import glob

entities = ["Premise1Conclusion", "Premise2Justification", "Collective", "Property", "pivot"]
components_amount = {}
components_words = {}
components_checked = {}

for ent in entities:
    components_amount[ent] = 0
    components_words[ent] = 0
#    components_checked[ent] = False

for f in glob.glob("./*.ann"):
    ff = open(f, "r")
    for ent in entities:
        components_checked[ent] = False
    for idx, line in enumerate(ff):
        parsed_line = line.replace("\n", "").split("\t")
        if len(parsed_line) > 1:
            component = (parsed_line[1].split(" "))[0]
            if component in entities:
                if not components_checked[component]:
                    components_amount[component] += 1
                    components_checked[component] = True
                components_words[component] += len(parsed_line[2].split())

print(components_amount)
print(components_words)
