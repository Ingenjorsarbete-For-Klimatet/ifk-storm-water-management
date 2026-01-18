
    
    
    
import rasterio
import matplotlib.pyplot as plt
from storm_water_management import postprocess, tif_preprocessing_utils
from dataclasses import dataclass

tif_file = "/home/chris/repos/data/results/trollhättan_depression_depth.tif"
@dataclass
class City:
    name: str
    longitudinal_position: float
    lateral_position: float
    distance_limit_calculation: float = 4000
    distance_limit_results: float = 1000
city = City("trollhättan", 58.28852773933211, 12.296952550648106)
x, y = tif_preprocessing_utils.get_sweref99_coordinate_from_wgs84(city.longitudinal_position, city.lateral_position)

cropped_output_tif_filename = postprocess.crop_tif(tif_file, x, y, city.distance_limit_results)
postprocess.write_geojson_polygons_from_tif_to_file(cropped_output_tif_filename)

with rasterio.open(cropped_output_tif_filename) as src:
    data = src.read(1)

plt.figure(figsize=(6, 6))
plt.imshow(data, cmap="viridis")
plt.colorbar(label="Värde")
plt.title("GeoTIFF")
plt.axis("off")
plt.show()
