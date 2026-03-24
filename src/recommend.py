def recommend_maintenance(row):
    recommendations = []

    if row["Tool wear [min]"] > 200:
        recommendations.append("Inspect or replace tool")

    if row["Torque [Nm]"] > 60:
        recommendations.append("Reduce load or inspect mechanical components")

    if row["temp_diff"] < 10:
        recommendations.append("Check cooling system")

    if row["Rotational speed [rpm]"] < 1200:
        recommendations.append("Inspect motor performance")

    if not recommendations:
        recommendations.append("No immediate maintenance needed")

    return recommendations