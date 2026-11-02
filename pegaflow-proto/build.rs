fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/engine.proto");
    println!("cargo:rerun-if-changed=proto/engine_v2.proto");

    tonic_prost_build::compile_protos("proto/engine.proto")?;
    tonic_prost_build::compile_protos("proto/engine_v2.proto")?;

    Ok(())
}
