-- Unit tests of cone beam ct fp preprocess
-- [1.06066017178f32,1.06066017178f32,1.76776695297f32,1.76776695297f32]
-- [1.76776695297f32,1.76776695297f32,1.06066017178f32,1.06066017178f32]
-- [0.5f32,-0.5f32,0.5f32,-0.5f32]
-- ==
-- compiled input {
--  [0.0f32, 0.78539816339f32]
--  2.0f32
--  2.0f32
--  2i32
-- }
-- output {
-- [2.0f32,2.0f32,2.0f32,2.0f32]
-- [0.5f32,0.5f32,-0.5f32,-0.5f32]
-- [0.5f32,-0.5f32,0.5f32,-0.5f32]
-- [0,2,1,3,4,6,5,7]
-- [0f32,0f32,0f32,0f32]
-- [-1.41421356237f32,-1.41421356237f32,-1.41421356237f32,-1.41421356237f32]
-- [-1.41421356237f32,-1.41421356237f32,-1.41421356237f32,-1.41421356237f32]
-- [1.06066017178f32,1.06066017178f32,1.76776695297f32,1.76776695297f32]
-- [1.76776695297f32,1.76776695297f32,1.06066017178f32,1.06066017178f32]
-- [0.5f32,-0.5f32,0.5f32,-0.5f32]
-- }

import "forwardprojection_cb"
open fplib

let main  (angles: []f32)
          (origin_source_dist: f32)
          (origin_detector_dist: f32)
          (detector_size: i32): ([]f32,[]f32,[]f32,[]i32,[]f32,[]f32,[]f32,[]f32,[]f32,[]f32) =
           let (x_small, y_small, z_small, _, projection_indexes) = preprocess angles origin_source_dist origin_detector_dist detector_size
           let (source_xsmall,detector_xsmall) = unzip x_small
           let (source_ysmall,detector_ysmall) = unzip y_small
           let (source_zsmall,detector_zsmall) = unzip z_small
           let (source_xsmall_x, source_xsmall_y, source_xsmall_z) = unzip3 source_xsmall
           let (source_ysmall_x, source_ysmall_y, source_ysmall_z) = unzip3 source_ysmall
           let (source_zsmall_x, source_zsmall_y, source_zsmall_z) = unzip3 source_zsmall
           let (detector_xsmall_x, detector_xsmall_y, detector_xsmall_z) = unzip3 detector_xsmall
           let (detector_ysmall_x, detector_ysmall_y, detector_ysmall_z) = unzip3 detector_ysmall
           let (detector_zsmall_x, detector_zsmall_y, detector_zsmall_z) = unzip3 detector_zsmall
           in (detector_ysmall_y, detector_ysmall_x, detector_ysmall_z,projection_indexes,source_zsmall_x, source_zsmall_y, source_zsmall_z,detector_zsmall_z, detector_zsmall_y, detector_zsmall_x)
