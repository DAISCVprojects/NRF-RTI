To render an object, please follow the following procedures, sample geometric object and material is provided
If you want to render your own images for diffrent geometric objects and matrials, Please refere to our paper and download the objects from their source

1. put the geometric object in /Geometric Model
2. put the material in /materials
3. Put the Light direction in /light
4. Give the path for the geometric object, material and light in the corresponding scripts

# Then run 'render.py' on cmd. 
# If the quality is not good(too dark , too bright, etc), you should check the parameters in 'sceneFile' or 'sceneTexture'

# we have provided two xml files, one is for rendering an object without explicilty using a texture (sceneFile) and the other is with texture(sceneTexture)
# each gemetric object of partcular matria type requires different parameters for rendering.
# we provided the parameters for each object in the comment section of 'sceneFile' and  'sceneTexture'
# You can improve the quality of rendered images by playing with those parameters

# Finally, We want to remind that we have noticed some images of our dataset have more dark regions for some light directions close to the objects surface .
# you can avoid this by inreasing the height 
