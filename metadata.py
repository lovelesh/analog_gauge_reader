from easygui import *
from pipeline_with_webcam_onnx_runtime import read_json_file, write_json_file
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

def save_meter_data(x, y):
    title = "Meter Metadata"
    msg = "Enter metadata for the Gauge"
    fieldNames = ["Name", "ID", "Start", "End", "Unit"]
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
                "id" : int(fieldValues[1]),
                "start" : int(fieldValues[2]),
                "end" : int(fieldValues[3]),
                "unit" : fieldValues[4],
                "center" : list((x, y)) 
        })
        # print(meter_config)
        return True

def read_metadata(metadata_file_path):
    dictionary = read_json_file(metadata_file_path)
    camera_id = dictionary["camera_id"]
    meter_configuration = dictionary["meter_config"]
    return camera_id, meter_configuration

def write_metadata(camera_id, meter_configuration, metadata_file_path):
    dictionary = { "camera_id": camera_id, "meter_config": meter_configuration}
    write_json_file(metadata_file_path, dictionary)