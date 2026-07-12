//! PegaFlow P/D Disaggregation Router
//!
//! Simple router that coordinates prefill (P) and decode (D) nodes.
//! Flow:
//! 1. Receive request
//! 2. Send to P node (max_tokens=1)
//! 3. Forward to D node (P response means KV is ready)
//!
//! `--pd-first-token` (variant A, for decode nodes that refuse ALL prompt
//! compute — GLM5.2 openinfer-D): P's single generated token is returned to
//! the client as part of the merged response AND appended to the token-id
//! prompt forwarded to D, so D's first forward is a true one-token decode.
//! P runs with `return_token_ids`; D is always driven through
//! `/v1/completions` with pre-tokenized ids (chat templates apply only on P),
//! and its SSE stream is re-shaped to the client's API. A D failure (e.g. the
//! strict no-prefill decode node rejecting a request whose remote KV never
//! landed) retries the whole P→D flow — content-addressed KV makes the P leg
//! idempotent.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Instant;

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use clap::Parser;
use log::{error, info};
use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_stream::StreamExt;

#[derive(Clone)]
struct RouterState {
    prefill_clients: Arc<Vec<Client>>,
    decode_clients: Arc<Vec<Client>>,
    prefill_urls: Arc<Vec<String>>,
    decode_urls: Arc<Vec<String>>,
    p_index: Arc<AtomicUsize>,
    d_index: Arc<AtomicUsize>,
    // Track in-flight requests per node
    p_inflight: Arc<Vec<AtomicUsize>>,
    d_inflight: Arc<Vec<AtomicUsize>>,
    // Variant A: forward P's first token into D's context (see module doc).
    pd_first_token: bool,
    // Full P->D flow retries on a D failure in variant A.
    pd_flow_retries: usize,
    // Model context limit: bounds the D leg's max_tokens when the client
    // sent none (both vLLM and openinfer default an absent max_tokens to 16,
    // which would silently truncate every open-ended chat request).
    pd_max_model_len: usize,
}

impl RouterState {
    fn new(prefill_endpoints: Vec<String>, decode_endpoints: Vec<String>) -> Self {
        let prefill_clients = prefill_endpoints
            .iter()
            .map(|_| {
                Client::builder()
                    .build()
                    .expect("Failed to build prefill client")
            })
            .collect();

        let decode_clients = decode_endpoints
            .iter()
            .map(|_| {
                Client::builder()
                    .build()
                    .expect("Failed to build decode client")
            })
            .collect();

        let p_inflight: Vec<AtomicUsize> = prefill_endpoints
            .iter()
            .map(|_| AtomicUsize::new(0))
            .collect();
        let d_inflight: Vec<AtomicUsize> = decode_endpoints
            .iter()
            .map(|_| AtomicUsize::new(0))
            .collect();

        Self {
            prefill_clients: Arc::new(prefill_clients),
            decode_clients: Arc::new(decode_clients),
            prefill_urls: Arc::new(prefill_endpoints),
            decode_urls: Arc::new(decode_endpoints),
            p_index: Arc::new(AtomicUsize::new(0)),
            d_index: Arc::new(AtomicUsize::new(0)),
            p_inflight: Arc::new(p_inflight),
            d_inflight: Arc::new(d_inflight),
            pd_first_token: false,
            pd_flow_retries: 1,
            pd_max_model_len: 32768,
        }
    }

    fn get_next_p(&self) -> (Client, String, usize) {
        let idx = self.p_index.fetch_add(1, Ordering::Relaxed);
        let idx = idx % self.prefill_clients.len();
        self.p_inflight[idx].fetch_add(1, Ordering::Relaxed);
        (
            self.prefill_clients[idx].clone(),
            self.prefill_urls[idx].clone(),
            idx,
        )
    }

    fn get_next_d(&self) -> (Client, String, usize) {
        let idx = self.d_index.fetch_add(1, Ordering::Relaxed);
        let idx = idx % self.decode_clients.len();
        self.d_inflight[idx].fetch_add(1, Ordering::Relaxed);
        (
            self.decode_clients[idx].clone(),
            self.decode_urls[idx].clone(),
            idx,
        )
    }

    fn finish_p(&self, idx: usize) {
        self.p_inflight[idx].fetch_sub(1, Ordering::Relaxed);
    }

    fn finish_d(&self, idx: usize) {
        self.d_inflight[idx].fetch_sub(1, Ordering::Relaxed);
    }

    fn get_inflight_summary(&self) -> String {
        let p_counts: Vec<usize> = self
            .p_inflight
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let d_counts: Vec<usize> = self
            .d_inflight
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        format!("P={:?} D={:?}", p_counts, d_counts)
    }
}

/// What the variant-A flow extracted from the P leg.
struct PrefillLeg {
    /// P's full prompt token ids (post chat template on the chat API).
    prompt_token_ids: Vec<u32>,
    /// The single generated token.
    t1_id: u32,
    /// Its detokenized text (chat: message.content; completions: text).
    t1_text: String,
    /// P's finish reason for that token ("stop" = EOS at t1: skip D).
    finish_reason: Option<String>,
    /// P's full response (the client envelope when D is skipped).
    response: Value,
}

/// A failed P->D attempt. `retryable` marks transient failures (network,
/// 5xx/429) — the whole flow may re-run because the P leg is idempotent
/// (same prompt, same content-addressed KV). Deterministic 4xx must not
/// retry: it would burn P compute on a guaranteed-identical failure.
struct FlowError {
    retryable: bool,
    msg: String,
}

impl FlowError {
    fn transient(msg: String) -> Self {
        Self {
            retryable: true,
            msg,
        }
    }

    fn from_status(status: StatusCode, msg: String) -> Self {
        Self {
            retryable: !status.is_client_error() || status == StatusCode::TOO_MANY_REQUESTS,
            msg,
        }
    }
}

/// One P->D attempt of the variant-A flow.
async fn pd_first_token_flow(
    state: &RouterState,
    body: &Value,
    api_path: &str,
    req_id: &str,
    arrive_time: Instant,
) -> Result<Response, FlowError> {
    let is_chat = api_path.ends_with("/chat/completions");
    let org_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let org_max_tokens = ["max_completion_tokens", "max_tokens"]
        .iter()
        .find_map(|k| body.get(*k).and_then(|v| v.as_u64()));

    // ---- P leg: prefill + exactly one generated token, with token ids ----
    let mut p_body = body.clone();
    p_body["max_tokens"] = json!(1);
    if p_body.get("max_completion_tokens").is_some() {
        p_body["max_completion_tokens"] = json!(1);
    }
    p_body["stream"] = json!(false);
    p_body["request_id"] = json!(req_id);
    p_body["return_token_ids"] = json!(true);
    // The appended-token contract needs t1 to exist even at instant EOS.
    p_body["min_tokens"] = json!(1);
    if let Some(obj) = p_body.as_object_mut() {
        obj.remove("stream_options");
    }

    let (p_client, p_url, p_idx) = state.get_next_p();
    let p_url = format!("{}{}", p_url, api_path);
    let p_response = p_client
        .post(&p_url)
        .header("X-Request-Id", req_id)
        .json(&p_body)
        .send()
        .await;
    state.finish_p(p_idx);
    let p_response = p_response.map_err(|e| FlowError::transient(format!("prefill request: {e}")))?;
    let p_status = p_response.status();
    let p_result: Value = p_response
        .json()
        .await
        .map_err(|e| FlowError::transient(format!("prefill response parse: {e}")))?;
    if !p_status.is_success() {
        return Err(FlowError::from_status(
            p_status,
            format!("prefill status {p_status}: {p_result}"),
        ));
    }
    let leg = extract_prefill_leg(p_result).map_err(FlowError::transient)?;
    info!(
        "prefill done: req={} P[{}] prompt_tokens={} t1={} latency={}ms",
        req_id,
        p_idx,
        leg.prompt_token_ids.len(),
        leg.t1_id,
        arrive_time.elapsed().as_millis()
    );

    // EOS at t1, or the client only asked for one token: P's response IS the
    // final answer — no decode leg.
    if leg.finish_reason.as_deref() == Some("stop") || org_max_tokens == Some(1) {
        return Ok(reshape_single_token_response(&leg, is_chat, org_stream));
    }

    // ---- D leg: pre-tokenized prompt + t1, always /v1/completions ----
    let mut d_body = body.clone();
    if let Some(obj) = d_body.as_object_mut() {
        obj.remove("messages");
        obj.remove("max_completion_tokens");
        obj.remove("echo");
        if is_chat {
            // Chat-only field shapes that /v1/completions rejects
            // (`logprobs` is a bool in chat, an int in completions).
            obj.remove("logprobs");
            obj.remove("top_logprobs");
            obj.remove("response_format");
            obj.remove("tools");
            obj.remove("tool_choice");
        }
    }
    let mut d_prompt = leg.prompt_token_ids.clone();
    d_prompt.push(leg.t1_id);
    d_body["prompt"] = json!(d_prompt);
    // An absent max_tokens must be made explicit: engines default it to 16,
    // silently truncating every open-ended request.
    let d_max = org_max_tokens.map(|max| max.saturating_sub(1).max(1)).unwrap_or_else(|| {
        (state.pd_max_model_len as u64)
            .saturating_sub(d_prompt.len() as u64)
            .max(1)
    });
    d_body["max_tokens"] = json!(d_max);
    if let Some(min_tokens) = body.get("min_tokens").and_then(|v| v.as_u64()) {
        d_body["min_tokens"] = json!(min_tokens.saturating_sub(1));
    }
    d_body["stream"] = json!(org_stream);
    d_body["request_id"] = json!(req_id);
    if org_stream {
        // Usage patching (t1 + P-side prompt length) needs the final counts.
        d_body["stream_options"] = json!({"include_usage": true});
    }

    let (d_client, d_url, d_idx) = state.get_next_d();
    let d_url = format!("{}/v1/completions", d_url);
    let d_response = match d_client
        .post(&d_url)
        .header("X-Request-Id", req_id)
        .json(&d_body)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            state.finish_d(d_idx);
            return Err(FlowError::transient(format!("decode request: {e}")));
        }
    };
    let d_status = d_response.status();
    if !d_status.is_success() {
        let detail = d_response.text().await.unwrap_or_default();
        state.finish_d(d_idx);
        return Err(FlowError::from_status(
            d_status,
            format!("decode status {d_status}: {detail}"),
        ));
    }

    if org_stream {
        Ok(stream_spliced_response(
            state.clone(),
            d_response,
            leg,
            is_chat,
            body.get("stream_options").cloned(),
            req_id.to_string(),
            d_idx,
            arrive_time,
        ))
    } else {
        let d_result: Result<Value, _> = d_response.json().await;
        state.finish_d(d_idx);
        let d_result =
            d_result.map_err(|e| FlowError::transient(format!("decode response parse: {e}")))?;
        info!(
            "done: req={} D[{}] total={}ms inflight=[{}]",
            req_id,
            d_idx,
            arrive_time.elapsed().as_millis(),
            state.get_inflight_summary()
        );
        Ok(merge_final_response(&leg, d_result, is_chat))
    }
}

/// Pull prompt ids + the single token out of P's chat/completions response.
fn extract_prefill_leg(p_result: Value) -> Result<PrefillLeg, String> {
    let choice = p_result
        .get("choices")
        .and_then(|c| c.get(0))
        .ok_or("prefill response has no choices")?;
    // Completions responses carry prompt_token_ids on the choice; chat
    // responses carry it at the top level.
    let prompt_token_ids: Vec<u32> = choice
        .get("prompt_token_ids")
        .or_else(|| p_result.get("prompt_token_ids"))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .ok_or("prefill response missing prompt_token_ids (P must support return_token_ids)")?;
    let token_ids: Vec<u32> = choice
        .get("token_ids")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .ok_or("prefill response missing token_ids")?;
    let t1_id = *token_ids
        .first()
        .ok_or("prefill generated no token (min_tokens=1 expected)")?;
    let t1_text = choice
        .get("text")
        .and_then(|v| v.as_str())
        .or_else(|| {
            choice
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(|v| v.as_str())
        })
        .unwrap_or_default()
        .to_string();
    let finish_reason = choice
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    Ok(PrefillLeg {
        prompt_token_ids,
        t1_id,
        t1_text,
        finish_reason,
        response: p_result,
    })
}

/// Strip the router-requested token-id fields (`return_token_ids`) from a
/// P response before it reaches the client — internal splice protocol, not
/// part of the public API the client called.
fn strip_internal_token_fields(mut response: Value) -> Value {
    if let Some(obj) = response.as_object_mut() {
        obj.remove("prompt_token_ids");
    }
    if let Some(choices) = response.get_mut("choices").and_then(|c| c.as_array_mut()) {
        for choice in choices {
            if let Some(obj) = choice.as_object_mut() {
                obj.remove("prompt_token_ids");
                obj.remove("token_ids");
            }
        }
    }
    response
}

/// The no-D-leg case: P's non-streaming response answers the client. A
/// streaming client gets it re-shaped as a minimal SSE exchange.
fn reshape_single_token_response(leg: &PrefillLeg, is_chat: bool, org_stream: bool) -> Response {
    if !org_stream {
        return (
            StatusCode::OK,
            Json(strip_internal_token_fields(leg.response.clone())),
        )
            .into_response();
    }
    let id = leg
        .response
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("pd-1")
        .to_string();
    let model = leg
        .response
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let finish = leg.finish_reason.clone().unwrap_or_else(|| "stop".into());
    let chunks = if is_chat {
        vec![
            chat_chunk(&id, &model, json!({"role": "assistant", "content": leg.t1_text}), None),
            chat_chunk(&id, &model, json!({}), Some(&finish)),
        ]
    } else {
        vec![completion_chunk(&id, &model, &leg.t1_text, Some(&finish))]
    };
    let mut sse = String::new();
    for chunk in chunks {
        sse.push_str(&format!("data: {chunk}\n\n"));
    }
    sse.push_str("data: [DONE]\n\n");
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .body(Body::from(sse))
        .unwrap()
}

fn chat_chunk(id: &str, model: &str, delta: Value, finish: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
}

fn completion_chunk(id: &str, model: &str, text: &str, finish: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "text_completion",
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish}],
    })
}

/// Non-streaming merge: t1 + D's completion, with usage covering both legs.
fn merge_final_response(leg: &PrefillLeg, d_result: Value, is_chat: bool) -> Response {
    let d_choice = &d_result["choices"][0];
    let d_text = d_choice["text"].as_str().unwrap_or_default();
    let finish = d_choice.get("finish_reason").cloned().unwrap_or(Value::Null);
    let d_completion_tokens = d_result["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    let usage = json!({
        "prompt_tokens": leg.prompt_token_ids.len(),
        "completion_tokens": d_completion_tokens + 1,
        "total_tokens": leg.prompt_token_ids.len() as u64 + d_completion_tokens + 1,
    });
    let mut out = strip_internal_token_fields(leg.response.clone());
    out["usage"] = usage;
    if is_chat {
        out["choices"][0]["message"]["content"] = json!(format!("{}{}", leg.t1_text, d_text));
    } else {
        out["choices"][0]["text"] = json!(format!("{}{}", leg.t1_text, d_text));
    }
    out["choices"][0]["finish_reason"] = finish;
    (StatusCode::OK, Json(out)).into_response()
}

/// Streaming splice: one synthetic chunk carrying t1, then D's completions
/// SSE re-shaped to the client's API (chat delta chunks when the client
/// spoke chat), with usage patched to cover both legs.
#[allow(clippy::too_many_arguments)]
fn stream_spliced_response(
    state: RouterState,
    d_response: reqwest::Response,
    leg: PrefillLeg,
    is_chat: bool,
    client_stream_options: Option<Value>,
    req_id: String,
    d_idx: usize,
    arrive_time: Instant,
) -> Response {
    let include_usage = client_stream_options
        .as_ref()
        .and_then(|o| o.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(100);
    tokio::spawn(async move {
        let id = leg
            .response
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("pd-1")
            .to_string();
        let model = leg
            .response
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let first = if is_chat {
            chat_chunk(&id, &model, json!({"role": "assistant", "content": leg.t1_text}), None)
        } else {
            completion_chunk(&id, &model, &leg.t1_text, None)
        };
        let _ = tx
            .send(Ok(format!("data: {first}\n\n").into()))
            .await;

        let mut buffer = String::new();
        let mut stream = Box::pin(d_response.bytes_stream());
        // t1 is already on the wire, so a mid-stream D failure cannot retry.
        // The client must still be able to tell "finished" from "truncated":
        // emit an error frame instead of a clean-looking EOF.
        let mut saw_done = false;
        'outer: while let Some(chunk) = stream.next().await {
            let Ok(bytes) = chunk else {
                error!("stream error: req={req_id}");
                break;
            };
            buffer.push_str(&String::from_utf8_lossy(&bytes));
            // SSE events are \n\n-delimited; hold the trailing partial event.
            while let Some(pos) = buffer.find("\n\n") {
                let event = buffer[..pos].to_string();
                buffer.drain(..pos + 2);
                let Some(data) = event
                    .lines()
                    .find_map(|line| line.strip_prefix("data: "))
                else {
                    continue;
                };
                if data.trim() == "[DONE]" {
                    saw_done = true;
                    let _ = tx.send(Ok("data: [DONE]\n\n".into())).await;
                    break 'outer;
                }
                let Ok(mut value) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                // Patch usage to cover both legs (P's prompt + t1).
                let is_usage_chunk = value.get("usage").is_some_and(|u| !u.is_null());
                if is_usage_chunk {
                    let d_completion =
                        value["usage"]["completion_tokens"].as_u64().unwrap_or(0);
                    value["usage"] = json!({
                        "prompt_tokens": leg.prompt_token_ids.len(),
                        "completion_tokens": d_completion + 1,
                        "total_tokens": leg.prompt_token_ids.len() as u64 + d_completion + 1,
                    });
                    if !include_usage {
                        // We forced include_usage on the D leg; a client that
                        // didn't ask for it must not see a usage-only chunk.
                        if value
                            .get("choices")
                            .and_then(|c| c.as_array())
                            .is_none_or(|c| c.is_empty())
                        {
                            continue;
                        }
                        value.as_object_mut().map(|o| o.remove("usage"));
                    }
                }
                let out = if is_chat {
                    let text = value["choices"][0]["text"].as_str().unwrap_or_default();
                    let finish = value["choices"][0]
                        .get("finish_reason")
                        .and_then(|v| v.as_str());
                    let delta = if text.is_empty() {
                        json!({})
                    } else {
                        json!({"content": text})
                    };
                    let mut chunk = chat_chunk(&id, &model, delta, finish);
                    if is_usage_chunk && include_usage {
                        chunk["usage"] = value["usage"].clone();
                        chunk["choices"] = json!([]);
                    }
                    chunk
                } else {
                    value
                };
                if tx
                    .send(Ok(format!("data: {out}\n\n").into()))
                    .await
                    .is_err()
                {
                    break 'outer;
                }
            }
        }
        if !saw_done {
            let err = json!({
                "error": {
                    "message": "decode stream ended before completion",
                    "type": "server_error",
                    "code": "pd_decode_interrupted",
                }
            });
            let _ = tx.send(Ok(format!("data: {err}\n\n").into())).await;
        }
        state.finish_d(d_idx);
        info!(
            "done (stream): req={} D[{}] total={}ms inflight=[{}]",
            req_id,
            d_idx,
            arrive_time.elapsed().as_millis(),
            state.get_inflight_summary()
        );
    });
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .body(Body::from_stream(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
        .unwrap()
}

async fn handle_completion(
    State(state): State<RouterState>,
    _headers: HeaderMap,
    Json(body): Json<Value>,
    api_path: &str,
) -> Response {
    let arrive_time = Instant::now();

    // Use existing request_id or generate new one
    let req_id = body
        .get("request_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    info!(
        "request arrived: req={} inflight=[{}]",
        req_id,
        state.get_inflight_summary()
    );

    if state.pd_first_token {
        // The splice takes choices[0] of a single prompt; anything else
        // would be silently mangled — refuse instead.
        if body.get("n").and_then(|v| v.as_u64()).is_some_and(|n| n > 1) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "P/D first-token routing does not support n > 1"})),
            )
                .into_response();
        }
        if body.get("prompt").and_then(|v| v.as_array()).is_some_and(|arr| {
            // A flat array of ints is ONE pre-tokenized prompt; anything
            // string-or-array-valued is a multi-prompt batch.
            arr.iter().any(|v| v.is_string() || v.is_array())
        }) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "P/D first-token routing does not support batched prompts"})),
            )
                .into_response();
        }
        let mut last_err = String::new();
        for attempt in 0..=state.pd_flow_retries {
            match pd_first_token_flow(&state, &body, api_path, &req_id, arrive_time).await {
                Ok(response) => return response,
                Err(err) => {
                    error!(
                        "P/D flow attempt {attempt} failed: req={req_id} retryable={} {}",
                        err.retryable, err.msg
                    );
                    last_err = err.msg;
                    if !err.retryable {
                        break;
                    }
                }
            }
        }
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": format!("P/D flow failed after retries: {last_err}")})),
        )
            .into_response();
    }

    // Save original values to restore for D request
    let org_max_tokens = body.get("max_tokens").cloned();
    let org_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let stream_options = body.get("stream_options").cloned();

    // Prepare P request (max_tokens=1, non-streaming)
    let mut p_body = body.clone();
    p_body["max_tokens"] = json!(1);
    // Chat clients send the modern `max_completion_tokens` field (OpenAI
    // deprecated `max_tokens` for chat); engines prefer it when both are
    // present, so it must be capped too or P runs the full decode.
    if p_body.get("max_completion_tokens").is_some() {
        p_body["max_completion_tokens"] = json!(1);
    }
    p_body["stream"] = json!(false);
    p_body["request_id"] = json!(req_id.clone());

    // Remove stream_options since stream=false
    p_body
        .as_object_mut()
        .map(|obj| obj.remove("stream_options"));

    // Ensure min_tokens <= max_tokens to avoid 400 from P node
    if let Some(min_tokens) = p_body.get("min_tokens").and_then(|v| v.as_i64()) {
        p_body["min_tokens"] = json!(min_tokens.min(1));
    } else {
        p_body["min_tokens"] = json!(0);
    }

    let (p_client, p_url, p_idx) = state.get_next_p();
    let p_url = format!("{}{}", p_url, api_path);

    // Send to P node
    let p_response = match p_client
        .post(&p_url)
        .header("X-Request-Id", &req_id)
        .json(&p_body)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            state.finish_p(p_idx);
            error!("P request failed: req={} error={}", req_id, e);
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": format!("Prefill node error: {}", e)})),
            )
                .into_response();
        }
    };

    let p_status = p_response.status();
    let p_result: Value = match p_response.json().await {
        Ok(v) => v,
        Err(e) => {
            state.finish_p(p_idx);
            error!("P response parse failed: req={} error={}", req_id, e);
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": format!("Prefill response error: {}", e)})),
            )
                .into_response();
        }
    };

    // P node finished
    state.finish_p(p_idx);

    if !p_status.is_success() {
        error!(
            "P error: req={} status={} body={:?}",
            req_id, p_status, p_result
        );
        return (p_status, Json(p_result)).into_response();
    }

    let prefill_latency = arrive_time.elapsed().as_millis();
    info!(
        "prefill done: req={} P[{}] latency={}ms inflight=[{}]",
        req_id,
        p_idx,
        prefill_latency,
        state.get_inflight_summary()
    );

    // Prepare D request (restore original settings)
    let mut d_body = body;
    if let Some(max_tokens) = org_max_tokens {
        d_body["max_tokens"] = max_tokens;
    }
    d_body["stream"] = json!(org_stream);
    d_body["request_id"] = json!(req_id.clone());
    if let Some(stream_opts) = stream_options {
        d_body["stream_options"] = stream_opts;
    }

    let (d_client, d_url, d_idx) = state.get_next_d();
    let d_url = format!("{}{}", d_url, api_path);

    if org_stream {
        // Streaming response
        let d_response = match d_client
            .post(&d_url)
            .header("X-Request-Id", &req_id)
            .json(&d_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                state.finish_d(d_idx);
                error!("D request failed: req={} error={}", req_id, e);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": format!("Decode node error: {}", e)})),
                )
                    .into_response();
            }
        };

        let stream = d_response.bytes_stream();
        let state_clone = state.clone();
        let stream = tokio_stream::wrappers::ReceiverStream::new({
            let (tx, rx) = tokio::sync::mpsc::channel(100);
            let req_id = req_id.clone();
            tokio::spawn(async move {
                let mut stream = Box::pin(stream);
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if tx.send(Ok::<_, std::io::Error>(bytes)).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            error!("Stream error: req={} error={}", req_id, e);
                            break;
                        }
                    }
                }
                // D node finished (stream ended)
                state_clone.finish_d(d_idx);
                let total = arrive_time.elapsed().as_millis();
                info!(
                    "done (stream): req={} D[{}] total={}ms inflight=[{}]",
                    req_id,
                    d_idx,
                    total,
                    state_clone.get_inflight_summary()
                );
            });
            rx
        });

        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/event-stream")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        // Non-streaming response
        let d_response = match d_client
            .post(&d_url)
            .header("X-Request-Id", &req_id)
            .json(&d_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                state.finish_d(d_idx);
                error!("D request failed: req={} error={}", req_id, e);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": format!("Decode node error: {}", e)})),
                )
                    .into_response();
            }
        };

        let d_status = d_response.status();
        let d_result: Value = match d_response.json().await {
            Ok(v) => v,
            Err(e) => {
                state.finish_d(d_idx);
                error!("D response parse failed: req={} error={}", req_id, e);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": format!("Decode response error: {}", e)})),
                )
                    .into_response();
            }
        };

        // D node finished
        state.finish_d(d_idx);
        let total = arrive_time.elapsed().as_millis();
        info!(
            "done: req={} D[{}] total={}ms inflight=[{}]",
            req_id,
            d_idx,
            total,
            state.get_inflight_summary()
        );

        (d_status, Json(d_result)).into_response()
    }
}

async fn chat_completions(
    state: State<RouterState>,
    headers: HeaderMap,
    body: Json<Value>,
) -> Response {
    handle_completion(state, headers, body, "/v1/chat/completions").await
}

async fn completions(state: State<RouterState>, headers: HeaderMap, body: Json<Value>) -> Response {
    handle_completion(state, headers, body, "/v1/completions").await
}

#[derive(Parser)]
#[command(name = "pegaflow-router")]
#[command(version)]
#[command(about = "PegaFlow P/D Disaggregation Router")]
struct Args {
    /// Host to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Prefill endpoints
    #[arg(long, required = true, num_args = 1..)]
    prefill: Vec<String>,

    /// Decode endpoints
    #[arg(long, required = true, num_args = 1..)]
    decode: Vec<String>,

    /// Variant A first-token forwarding: return P's single token to the
    /// client and append it (as a token id) to the prompt D receives, so a
    /// strict no-prefill decode node's first forward is a one-token decode.
    /// Requires P to support `return_token_ids` and D to accept token-id
    /// prompts on /v1/completions.
    #[arg(long, default_value_t = false)]
    pd_first_token: bool,

    /// Full P->D flow retries when the D leg fails in --pd-first-token mode
    /// (a strict decode node rejects when the remote KV never lands).
    #[arg(long, default_value_t = 1)]
    pd_flow_retries: usize,

    /// Model context limit, bounding the D leg's max_tokens when the client
    /// sent none (engines default an absent max_tokens to 16).
    #[arg(long, default_value_t = 32768)]
    pd_max_model_len: usize,
}

#[tokio::main]
async fn main() {
    pegaflow_common::logging::init_stderr("info");
    info!("Starting pegaflow-router v{}", env!("CARGO_PKG_VERSION"));

    let args = Args::parse();

    let mut state = RouterState::new(args.prefill.clone(), args.decode.clone());
    state.pd_first_token = args.pd_first_token;
    state.pd_flow_retries = args.pd_flow_retries;
    state.pd_max_model_len = args.pd_max_model_len;
    if state.pd_first_token {
        info!(
            "P/D first-token forwarding on (flow retries: {})",
            state.pd_flow_retries
        );
    }

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    info!("Starting on {}", addr);
    info!("Prefill nodes: {:?}", args.prefill);
    info!("Decode nodes: {:?}", args.decode);

    let listener = TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
