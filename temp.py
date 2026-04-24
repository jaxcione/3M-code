import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

T_optimized=286.09 #kelvin this is the avg optimal temp
baseline=(T_optimized-273.15)*9/5+32 #converting to F
directory=r"3M\optimize\tempdata" #where all the data is stored 
file_order=(["data.csv"] +[f"data ({i}).csv" for i in range(1, 12)] ) #just how the files are ordered it goes from data.csv to data(11).csv
months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"] #number of months in an array
all_data = {} #storing this in a library

#going through each file and taking the value column which is jsut the thrid element for each row
for month_idx, filename in enumerate(file_order):
    filepath=os.path.join(directory, filename)
    if not os.path.exists(filepath): #skipping if the file doesnt even exist
        continue
    df = pd.read_csv(filepath, comment="#")
    df.columns = df.columns.str.strip()#removing any whitespace

    for _, row in df.iterrows():
        try:
            state = row["Name"].strip() #getting the name
            avg_temp_f = float(row["Value"]) #getting associated avg monthly temp value
        except (ValueError, TypeError):
            continue

        if state not in all_data: #if the data wasnt there prior store it with 12 empty values then fill the temp for that month there
            all_data[state] = [None]*12
        all_data[state][month_idx] = avg_temp_f

states = sorted(all_data.keys()) #sorting it based on temps
diff_data = {} #difference of data

for state in states: #going through each state
    monthly = all_data[state]
    diffs = [] 
    for t in monthly:
        if t is not None:
            diffs.append(abs((t - baseline))) #finding the differnce 
        else:
            diffs.append(None)
    diff_data[state] = diffs

avg_abs_diff = {} #storing the differnce  

for state, diffs in diff_data.items(): 
    values = []
    for d in diffs:
        if d is not None:
            values.append(abs(d))
    avg_abs_diff[state] = np.nanmean(values) #storing the abs value of the avg diff of each state

best_states = sorted(avg_abs_diff, key=avg_abs_diff.get)[:4] #getting the lowest 4 states

#plotting--------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 9)) #only one fig
colors = cm.tab20(np.linspace(0, 1, len(states)))
highlight_colors = ["red", "crimson", "orangered", "tomato"] #colors that will be used for each of the four states

x = np.arange(12) #for the twelve months 
for i, state in enumerate(states):
    diffs = diff_data[state]
    y = []
    for d in diffs:
        if d is not None:
            y.append(d)
        else:
            y.append(float("nan"))
    if state in best_states:
        rank = best_states.index(state) #ranking the best states
        ax.plot(x, y, marker="o", markersize=6, linewidth=3,
                color=highlight_colors[rank],
                label=f"#{rank+1} {state} ({avg_abs_diff[state]:.1f}°F)", zorder=5)
    else:
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.4,
                color=colors[i], label=state, alpha=0.85)

ax.axhline(0, color="black", linewidth=1.5, linestyle="--",
           label=f"Baseline ({T_optimized} K / {baseline} °F)")

ax.set_xticks(x)
ax.set_xticklabels(months, fontsize=11)
ax.set_ylabel("Temperature difference from baseline (°F)", fontsize=12)
ax.set_title(f"Monthly avg absolute temperature difference from {T_optimized} K ({baseline} °F)\n""by U.S. state — 2025", fontsize=14, fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend( loc="upper left",bbox_to_anchor=(1.01, 1),fontsize=7.5,ncol=2, frameon=False,title="State",title_fontsize=9)

plt.tight_layout()
output_path = os.path.join(directory, "temp_diff_by_state.png") #change to ur own directory or whatnot
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()