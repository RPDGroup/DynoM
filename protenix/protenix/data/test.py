import concurrent.futures

# 处理单个样本的函数
def process_sample(data, atom_array, data_error_message, runner,config):
    try:
        if len(data_error_message) > 0:
            logger.info(data_error_message)
            with open(
                opjoin(runner.error_dir, f"{data['sample_name']}.txt"),
                "w",
            ) as f:
                f.write(data_error_message)
            return None  # 出现错误则跳过该样本

        sample_name = data["sample_name"]
        logger.info(
            (
                f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] {sample_name}: "
            )
        )
        new_configs = update_inference_configs(configs, data["N_token"].item())
        runner.update_model_configs(new_configs)

        if configs.only_encoder:
            s_data, pair_data = runner.predict(data)
            np.save(f"{configs.dump_dir}/{data['sample_name']}_single_repr_recycle{configs.model.N_cycle}.npy",s_data)
            np.save(f"{configs.dump_dir}/{data['sample_name']}_pair_repr_recycle{configs.model.N_cycle}.npy", pair_data)
            continue
        else:
            prediction = runner.predict(data)

        
        runner.dumper.dump(
            dataset_name="",
            pdb_id=sample_name,
            seed=seed,
            pred_dict=prediction,
            atom_array=atom_array,
            entity_poly_type=data["entity_poly_type"],
        )

        logger.info(
            f"[Rank {DIST_WRAPPER.rank}] {data['sample_name']} succeeded.\n"
        )
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        error_message = f"[Rank {DIST_WRAPPER.rank}] {data['sample_name']} {e}:\n{traceback.format_exc()}"
        logger.info(error_message)
        with open(opjoin(runner.error_dir, f"{data['sample_name']}.txt"), "w") as f:
            f.write(error_message)
        return False


def process_batch(batch, runner):
    # 使用 ProcessPoolExecutor 来并行处理每个样本
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for data, atom_array, data_error_message in batch:
            futures.append(executor.submit(process_sample, data, atom_array, data_error_message, runner))

        # 等待所有任务完成
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # 返回处理结果
    return results


# 主代码段
for batch in dataloader:
    # 批次数据并行处理
    process_batch(batch, runner)
