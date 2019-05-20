-- Unit tests of cone beam ct implementation
-- ==
-- compiled input {
--  0.0f32
--  1.0f32
--  2.0f32
--  2i32
-- }
-- output { [0.5f32, -0.5f32, 0.5f32, -0.5f32] }

import "forwardprojection_cb"
open fplib

let main  (sin : f32)
           (cos : f32)
           (origin_detector_dist: f32)
           (detector_size : i32): []f32 =
           let (xyz,i) = unzip2 (find_detector_points sin cos origin_detector_dist detector_size)
           let (x,y,z) = unzip3 xyz
           in z
