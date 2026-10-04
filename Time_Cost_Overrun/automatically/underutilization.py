"""Part 3 - UNDER-UTILIZATION risk. Target: under_utilization_pct > 1%
(6.2% of 13,119 works). Same ANN+SVM architecture as time_overrun_model.py.
"""
from Time_Cost_Overrun.rates.model.time_overrun_model import train_and_evaluate

if __name__ == "__main__":
    print("*** Part 3: UNDER-UTILIZATION ***\n")
    results, df = train_and_evaluate(label_col="underutilization_label")
    print(f"ANN: {(df['segment_route']=='ANN').sum()} works | SVM: {(df['segment_route']=='SVM').sum()} works\n")
    for name, r in results.items():
        print(f"{name}: train={r['n_train']} test={r['n_test']} AUC={r['AUC']:.3f} accuracy={r['accuracy']:.3f}")