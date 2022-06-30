from glob import glob

collective = open("test_results_collective", 'w')
proper = open("test_results_property", 'w')
pivot = open("test_results_pivot", 'w')
conclusion = open("test_results_conclusion", 'w')
justification = open("test_results_justification", 'w')
argumentative = open("test_results_argumentative", 'w')
type_justification = open("test_results_type_justification", "w")
type_conclusion = open("test_results_type_conclusion", "w")


for filename in glob("./test_results_coling/*"):
    filename_splitted = filename.split("/")[2].split("_")
    print(filename_splitted)
    if filename_splitted[1] == "test":
        lr = filename_splitted[2]
        modelname = filename_splitted[3]
        component = filename_splitted[4]
        f = open(filename, 'r')
        to_write = ""
        for line in f:
                line_splitted = line.split(",")
                acc = line_splitted[0]
                f1 = line_splitted[1]
                precision = line_splitted[2]
                recall = line_splitted[3]

                to_write = "{},{},{},{},{},{}\n".format(modelname, lr, acc, f1, precision, recall)
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
            if filename_splitted[7] == "Premise1Conclusion":
                type_conclusion.write(to_write)
            elif filename_splitted[7] == "Premise2Justification":
                type_justification.write(to_write)
   
collective.close()
proper.close()
pivot.close()
conclusion.close()
justification.close()
argumentative.close() 
