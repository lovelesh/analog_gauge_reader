from easygui import *

# Gauge metadata

meter_config = []
# meter_config = [{
#             "name" : "boiler pressure",
#             "start" : 0,
#             "end" : 800,
#             "unit" : "kPa",
#             "center" : [901, 551]
# }, {
#             "name" : "tank pressure",
#             "start" : 0,
#             "end" : 200,
#             "unit" : "psi",
#             "center" : [949, 242]
# }, {
#             "name" : "reservoir pressure",
#             "start" : 0,
#             "end" : 400,
#             "unit" : "psi",
#             "center" : [568, 494]
# }]

def get_gauge_details(box):
    for index in range(len(meter_config)):
        if (meter_config[index]['center'][0] > int(box[0]) and 
            meter_config[index]['center'][0] < int(box[2]) and
            meter_config[index]['center'][1] > int(box[1]) and
            meter_config[index]['center'][1] < int(box[3])):
            return index
    
    print("no Gauge details found")
    raise Exception("no Gauge details found")

def save_metadata(x, y):
    title = "Meter Metadata"
    msg = "Enter metadata for the gauge"
    fieldNames = ["Name", "Start", "End", "Unit"]
    fieldValues = []
    fieldValues = multenterbox(msg, title, fieldNames)
    print(f"received info: {fieldValues}")
    if fieldValues == None:
        return False
    errmsg = ""
    for i in range(len(fieldNames)):
        if fieldValues[i].strip == "" :
            errmsg = errmsg + ("%s is required\n" % fieldNames[i])
    if errmsg == "":
        # copy fieldValues to meter_config variable
        meter_config.append({
                "name" : fieldValues[0],
                "start" : int(fieldValues[1]),
                "end" : int(fieldValues[2]),
                "unit" : fieldValues[3],
                "center" : list((x, y)) 
        })
        # print(meter_config)
        return True
      