from glob import glob
import statistics

collective = open("test_results_collective", 'w')
proper = open("test_results_property", 'w')
pivot = open("test_results_pivot", 'w')
conclusion = open("test_results_conclusion", 'w')
justification = open("test_results_justification", 'w')
argumentative = open("test_results_argumentative", 'w')
type_justification = open("test_results_type_justification", "w")
type_conclusion = open("test_results_type_conclusion", "w")


averages = {}
for filename in glob("./final_version_test_results_type_no_info/*"):
    filename_splitted = filename.split("/")[2].split("_")
    print(filename_splitted)
    if filename_splitted[1] == "test":
        lr = filename_splitted[2]
        modelname = filename_splitted[3]
        batch_size = filename_splitted[4]
        rep = filename_splitted[5]
        component = filename_splitted[6]
        f = open(filename, 'r')
        to_write = ""
        key = "{}_{}_{}_{}".format(component, modelname, lr, batch_size)
        if component == "Type":
            key = "{}_{}_{}_{}_{}".format(component, modelname, lr, batch_size, filename_splitted[9])
        for line in f:
                line_splitted = line.split(",")
                acc = line_splitted[0]
                precision = line_splitted[1]
                recall = line_splitted[2]
                f1_bin = line_splitted[3]

                if key not in averages:
                    averages[key] = [0.0, 0.0, 0.0, 0.0, 0, []]
                to_update = averages[key]
                to_update[0] += float(acc)
                to_update[1] += float(precision)
                to_update[2] += float(recall)
                to_update[3] += float(f1_bin.replace("\n", ""))
                to_update[4] += 1
                to_update[5].append(float(f1_bin.replace("\n", "")))
                averages[key] = to_update
                break

for k in averages:
    splitted = k.split("_")
    component = splitted[0]
    modelname = splitted[1]
    lr = splitted[2]
    if component == "Type":
        premise = splitted[4]
    values = averages[k]
    acc = values[0] / values [4]
    precision = values[1] / values[4]
    recall = values[2] / values[4]
    f1_minority = values[3] / values[4]
    if len(values[5]) > 1:
        stdev = statistics.stdev(values[5])
    else:
        stdev = 0
    to_write = "{},{},{},{},{},{},{}\n".format(modelname, lr, acc, precision, recall, f1_minority, stdev)
    if component == "Collective":
        collective.write(to_write)
    elif component == "Property":
        proper.write(to_write)
    elif component == "pivot":
        pivot.write(to_write)
    elif component == "Premise1Conclusion":
        conclusion.write(to_write)
    elif component == "Premise2Justification":
        justification.write(to_write)
    elif component == "NonArgumentative":
        argumentative.write(to_write)
    elif component == "Type":
        if premise == "Premise1Conclusion":
            type_conclusion.write(to_write)
        elif premise == "Premise2Justification":
            type_justification.write(to_write)
   
collective.close()
proper.close()
pivot.close()
conclusion.close()
justification.close()
argumentative.close()
type_conclusion.close()
type_justification.close()
