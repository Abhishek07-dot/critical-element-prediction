import ee
import pandas as pd

ee.Initialize(project='prediction-488105')

# DEM stack
dem = ee.Image("USGS/SRTMGL1_003")
terrain = ee.Terrain.products(dem)

slope = terrain.select("slope")
aspect = terrain.select("aspect")

curvature = dem.convolve(
    ee.Kernel.laplacian8()
).rename("curvature")

twi = slope.expression(
    "log((100 + 1) / (tan(slope * 0.01745) + 0.001))",
    {"slope": slope}
).rename("twi")

stack = dem.rename("elevation") \
    .addBands(slope.rename("slope")) \
    .addBands(aspect.rename("aspect")) \
    .addBands(curvature) \
    .addBands(twi)

# load original dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\Lithium_Statistics (1).csv")

results = []

for i,row in df.iterrows():

    point = ee.Geometry.Point(row["long"], row["lat"])

    sample = stack.sample(region=point,scale=30).first()

    props = sample.getInfo()["properties"]

    results.append({
        "lat":row["lat"],
        "long":row["long"],
        "elevation":props["elevation"],
        "slope":props["slope"],
        "aspect":props["aspect"],
        "curvature":props["curvature"],
        "twi":props["twi"]
    })

    if i % 200 == 0:
        print("Processed",i)

terrain_df = pd.DataFrame(results)

terrain_df.to_csv("terrain_variables.csv",index=False)

print("Terrain CSV created")