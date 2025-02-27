function_name = "_pop_waiting_trial_id"

for func, data in stats.stats.items():
    if function_name in func[2]:  # 関数名が一致するものを探す
        ncalls, tottime, percall, cumtime, percall_cum = data
        print(f"関数: {func[2]}")
        print(f"  呼び出し回数: {ncalls}")
        print(f"  合計実行時間 (tottime): {tottime:.6f} 秒")
        print(f"  累積時間 (cumtime): {cumtime:.6f} 秒")
        break
