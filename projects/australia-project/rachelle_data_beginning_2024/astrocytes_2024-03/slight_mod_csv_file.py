import pandas as pd


df = pd.read_csv("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/g_shared/shared/alberto/projects/spacem-reports/australia-project/2024/all_astrocytes/spacem-cells/SC_analysis_results/treatment_Cytokines_markers.csv")

df_cytokines = df.loc[df["scores"]>0]
df_cytokines.to_csv("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/g_shared/shared/alberto/projects/spacem-reports/australia-project/2024/all_astrocytes/spacem-cells/SC_analysis_results/cytokines_relevant_markers.csv", index=False)

df_control = df.loc[df["scores"]<0]
df_control["treatment"] = "Control"
df_control["scores"] = df_control["scores"].abs()
# sort descending according to scores:
df_control.sort_values("scores", ascending=False, inplace=True)
df_control.to_csv("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/g_shared/shared/alberto/projects/spacem-reports/australia-project/2024/all_astrocytes/spacem-cells/SC_analysis_results/control_relevant_markers.csv", index=False)
