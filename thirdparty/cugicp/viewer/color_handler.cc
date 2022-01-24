#include <algorithm>
#include <iostream>

#include <viewer/color_handler.h>

namespace perl_registration {

bool PointCloudColorHandlerIntensityMap::getColor(
    vtkSmartPointer<vtkDataArray>& scalars) const {
  if (!capable_ || !cloud_) {
    return false;
  }

  if (!scalars) scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
  scalars->SetNumberOfComponents(3);

  vtkIdType nr_points = cloud_->points.size();
  reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))
      ->SetNumberOfTuples(nr_points);

  unsigned char* colors = new unsigned char[nr_points * 3];
  double r, g, b;

  // Color every point
  for (vtkIdType cp = 0; cp < nr_points; ++cp) {
    float cuvrature = cloud_->points[cp].intensity;
    if (std::isnan(cuvrature)) cuvrature = max_;
    int intensity = std::round(std::max(
        std::min(((cuvrature - min_) / (max_ - min_) * 255.0), 255.0), 0.0));
    colors[cp * 3 + 0] =
        static_cast<unsigned char>(viridis_color_map(intensity, 0) * 255);
    colors[cp * 3 + 1] =
        static_cast<unsigned char>(viridis_color_map(intensity, 1) * 255);
    colors[cp * 3 + 2] =
        static_cast<unsigned char>(viridis_color_map(intensity, 2) * 255);
  }
  reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))
      ->SetArray(colors, 3 * nr_points, 0,
                 vtkUnsignedCharArray::VTK_DATA_ARRAY_DELETE);
  return (true);
}

}  // namespace perl_registration
