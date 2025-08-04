To render an object, please follow the following procedures. A sample geometric object and material are provided in this directory.
If you want to render your own images for different geometric objects and materials, please refer to our paper and download the objects from their source

1. Put the geometric object in /Geometric Model
2. Put the material in /materials
3. Put the Light direction in /light
4. Give the path for the geometric object, material, and light in the corresponding scripts

# Then run 'render.py' on the command line. 
# If the quality is not good(too dark, too bright, etc), you should check the parameters in 'sceneFile' or 'sceneTexture'

# We have provided two XML files, one is for rendering an object without explicitly using a texture (sceneFile), and the other is with a  texture(sceneTexture)
# Each geometric object of a particular material type requires different parameters for rendering.
# We provided the parameters for each object in the comment section of 'sceneFile' and  'sceneTexture'
# You can improve the quality of rendered images by playing with those parameters

# Finally, we want to remind that we have noticed some images of our dataset have more dark regions for some light directions close to the object's surface.
# You can avoid this by increasing the height 
