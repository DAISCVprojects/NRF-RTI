import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mitsuba as mi

mi.set_variant("scalar_rgb")

def lightPosition():
    # read light direction file
    df = pd.read_csv("/home/shambel/RTI_gray/SynthMLICs/Material2_320/light_test", names=[0, 1, 2])
    light_array = df.to_numpy()
    return light_array

def updateLightPosition(position):
    origin = position
    target = [0.0, 0.0, 0.0] # same as emitter target in the scene file
    up = [0, 1, 0]            # same as emitter up in the scene file
    transform_mat = mi.Transform4f.look_at(origin=origin, target=target, up=up)
    return transform_mat


# loding the scene file
scene = mi.load_file("sceneFile.xml")
#retrieving the parameters
params = mi.traverse(scene)

# accessing light direction file
light_array = lightPosition()

# rendering the scene for each light direction
print("Rendering the scene.....Please wait")
for pos in range(len(light_array)):
    transform_mat = updateLightPosition(list(light_array[pos]))
    # update transform matrix
    params['DirectionalEmitter.to_world'] = transform_mat
    # update the scene parameter
    params.update();
    
    # render the scene
    image = mi.render(scene)
    # writing the image to disk
    mi.util.write_bitmap('/home/shambel/RTI_gray/SynthMLICs/Dataset/test/' + 'image' + str(pos) + '.jpeg', image , write_async=True)
print("Rendering has completed")

    

    
