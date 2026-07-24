use super::*;

#[test]
fn python_instances_keep_legacy_multi_context_semantics() {
    let mut registry = CudaTensorRegistry::empty();

    for context_key in ["instance-a:tp0:pp0:dev0", "instance-a:tp1:pp0:dev1"] {
        registry
            .register_layers(context_key, "instance-a", 0, Vec::new(), None)
            .expect("Python instance may register multiple contexts");
    }

    assert_eq!(registry.contexts.len(), 2);
}

#[test]
fn native_registration_does_not_mix_with_python_contexts() {
    let mut registry = CudaTensorRegistry::empty();
    registry
        .register_layers("instance-a:tp0:pp0:dev0", "instance-a", 0, Vec::new(), None)
        .expect("register Python context");

    let err = match registry.register_layers(
        "instance-a:tp1:pp0:dev0",
        "instance-a",
        0,
        Vec::new(),
        Some((-1, 1)),
    ) {
        Err(err) => err,
        Ok(_) => panic!("native registration must fail before importing the invalid fd"),
    };
    let message = Python::attach(|py| err.value(py).to_string());
    assert!(message.contains("already has Python contexts"));
}

#[tokio::test]
async fn registry_cleanup_blocks_new_work_and_drains_in_flight_work() {
    let mut state = CudaTensorRegistry::empty();
    let registration = Arc::new(VmmRegistration::stub("native-a", 0));
    let mut context = ContextState::new(0);
    context.native_registration = Some(Arc::clone(&registration));
    state
        .contexts
        .insert("native-a:tp0:pp0:dev0".to_string(), context);
    state
        .native_instances
        .insert("native-a".to_string(), registration);
    let registry = RegistryHandle::spawn(state);
    let operation = registry
        .acquire_registration("native-a".to_string(), 0)
        .await
        .expect("acquire native operation")
        .expect("native registration");

    let cleanup = tokio::spawn({
        let registry = registry.clone();
        async move { registry.drop_instance("native-a".to_string()).await }
    });
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            match registry
                .acquire_registration("native-a".to_string(), 0)
                .await
            {
                Err(message) if message.contains("closing") => break,
                Ok(Some(guard)) => drop(guard),
                Ok(None) => panic!("native registration disappeared but context remained"),
                Err(message) => panic!("unexpected native close error: {message}"),
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("cleanup must remove the registration");
    assert!(
        !cleanup.is_finished(),
        "in-flight operation must block cleanup"
    );

    drop(operation);
    let cleanup = tokio::time::timeout(std::time::Duration::from_secs(1), cleanup)
        .await
        .expect("cleanup must drain after operation")
        .expect("cleanup task");
    registry.finish_cleanup(cleanup).await;
}

#[tokio::test]
async fn clear_waits_for_registration_to_finish() {
    let registry = RegistryHandle::spawn(CudaTensorRegistry::empty());
    let (_, _, registration_permit) = registry
        .register_layers(
            "instance-a:tp0:pp0:dev0".to_string(),
            "instance-a".to_string(),
            0,
            Vec::new(),
            None,
        )
        .await
        .expect("register context");
    let clear = tokio::spawn({
        let registry = registry.clone();
        async move { registry.clear().await }
    });

    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while Arc::clone(&registry.registration_barrier)
            .try_read_owned()
            .is_ok()
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("clear must queue for the registration barrier");
    let next_registration = tokio::spawn({
        let registry = registry.clone();
        async move {
            registry
                .register_layers(
                    "instance-b:tp0:pp0:dev0".to_string(),
                    "instance-b".to_string(),
                    0,
                    Vec::new(),
                    None,
                )
                .await
        }
    });
    assert!(
        !clear.is_finished(),
        "clear must not cross an in-progress registration"
    );
    assert!(
        !next_registration.is_finished(),
        "new registration must queue behind clear"
    );

    drop(registration_permit);
    let cleanup = tokio::time::timeout(std::time::Duration::from_secs(1), clear)
        .await
        .expect("clear must continue after registration")
        .expect("clear task");
    assert!(
        !next_registration.is_finished(),
        "clear must keep the barrier through engine cleanup"
    );
    registry.finish_cleanup(cleanup).await;
    tokio::time::timeout(std::time::Duration::from_secs(1), next_registration)
        .await
        .expect("next registration must continue after cleanup")
        .expect("registration task")
        .expect("register next context");
}
