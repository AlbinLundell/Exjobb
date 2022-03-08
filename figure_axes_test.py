# Code that creates a figure illustrating the loss of data.

import matplotlib.pyplot as plt
import DF_number_of_DP


df = DF_number_of_DP.df_data(kurs = "FYSFYS01a", datum = "11-15")
def datapoints_radius(df):
    """
    DF      Variable    Antal
    0       Betyg       XXX
    1       Frånvaro    YYY
    ...     ...         ...
    """
    factor_list = []
    total_number_DP = df['Antal'][0]
    for i in range(len(df)):
        current_DP_number = df['Antal'][i]
        factor = current_DP_number/total_number_DP      # vi utgår från betyg, därför samma
        factor_list.append(factor)                      # add the factor to the list
    for i in range(len(factor_list)):
        factor_list[i] = factor_list[i]*0.2
            # Måste multiplicera med en ANNAN faktor för att bestämma radien mer snyggt. # Nu * 0.1, kanske exponentiellt eller enligt någon annan skala

    return factor_list

radius_size_list = datapoints_radius(df)
print(radius_size_list)

fig = plt.figure(figsize=(16,8))
ax = plt.axes()             # frameon=False
ax.set_aspect(1)        # x-axis twice as long as y-axis
ax.set_xlim([0,2])      # increase the x-axis coordinate two 2

# Create a text font to be used for all circle.


# plt.Circle((x-location, y-location), radiussize, color, alpha = transparency percentage)

# Circle Betyg
circleBetyg = plt.Circle((0.3, 0.5), radius_size_list[0], color='y', alpha = 0.5)
plt.text(0.3, 0.5 - 0.03, f"Betyg\n{df['Antal'][0]}", ha = "center")
ax.add_patch(circleBetyg)

# Circle Frånvaro
circleFranvaro = plt.Circle((0.8, 0.7), radius_size_list[1], color='b', alpha = 0.5)
plt.text(0.8, 0.7 - 0.03, f"Frånvaro\n{df['Antal'][1]}", ha = "center")
ax.add_patch(circleFranvaro)

# Circles Diagnos 1+2
circleDiagnos = plt.Circle((1.3, 0.7), radius_size_list[2], color='g', alpha = 0.5)
plt.text(1.3, 0.7 - 0.03, f"Diagnos\n{df['Antal'][2]}", ha = "center")
ax.add_patch(circleDiagnos)
# The posistion of the circles below depend on the radius before
circleDiagnosPass1 = plt.Circle((1.3 - radius_size_list[2]*0.9, 0.7 - radius_size_list[2] - 0.2), radius_size_list[3] , color = 'm', alpha = 0.5)
plt.text(1.3 - radius_size_list[2]*0.9, 0.7 - radius_size_list[2] - 0.2 - 0.03, f"Diagnos P-F\n{df['Antal'][3]}", ha = "center")
ax.add_patch(circleDiagnosPass1)
circleDiagnosE1 = plt.Circle((1.3 + radius_size_list[2]*0.9, 0.7 - radius_size_list[2] - 0.2), radius_size_list[4], color = 'r', alpha = 0.5)
plt.text(1.3 + radius_size_list[2]*0.9, 0.7 - radius_size_list[2] - 0.2 - radius_size_list[4] - 0.07, f"Diagnos E-F\n{df['Antal'][4]}", ha = "center")
ax.add_patch(circleDiagnosE1)

# Circles Canvas 1+2
circleCanvas = plt.Circle((1.7, 0.7), radius_size_list[5], color='g', alpha = 0.5)
plt.text(1.7, 0.7 - 0.03, f"Canvas\n{df['Antal'][5]}", ha = "center")
ax.add_patch(circleCanvas)
# The posistion of the circles below depend on the radius before
circleCanvasPass1 = plt.Circle((1.7 - radius_size_list[5], 0.7 - radius_size_list[5] - 0.12), radius_size_list[6] , color = 'm', alpha = 0.5)
plt.text(1.7 - radius_size_list[5], 0.7 - radius_size_list[5] - 0.12 - 0.03, f"Canvas P-F\n{df['Antal'][6]}", ha = "center")
ax.add_patch(circleCanvasPass1)

circleCanvasE1 = plt.Circle((1.7 + radius_size_list[5], 0.7 - radius_size_list[5] - 0.12), radius_size_list[7], color = 'r', alpha = 0.5)
plt.text(1.7 + radius_size_list[5], 0.7 - radius_size_list[5] - 0.12 - radius_size_list[7] - 0.07, f"Canvas E-F\n{df['Antal'][7]}", ha = "center")
ax.add_patch(circleCanvasE1)


plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)         # Remove the axes lines.
plt.title("Number of datasamples (students) when new variables introduced, Fysik 1a")

# --------------- To do list , questionsmarks ---------------
# Have a bright color of the circle, alpha change this, change it even more?
# Arrows?
# Do I want to have a frame? Perhaps a thicker frame?
# Titel?
# Vilken kurs vi snackar om?
# Vilket Canvas datum?
# Ändra färger

plt.show()
