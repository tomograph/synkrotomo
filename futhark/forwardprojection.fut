-- ==
-- input@../data/fpinputf32rad64
-- input@../data/fpinputf32rad128
-- input@../data/fpinputf32rad256
-- input@../data/fpinputf32rad512
-- input@../data/fpinputf32rad1024
-- input@../data/fpinputf32rad1500
-- input@../data/fpinputf32rad2000
-- input@../data/fpinputf32rad2048
-- input@../data/fpinputf32rad2500
-- input@../data/fpinputf32rad3000
-- input@../data/fpinputf32rad3500
-- input@../data/fpinputf32rad4000
-- input@../data/fpinputf32rad4096
-- input@../data/fpinputf32rad_angles_fixed64
-- input@../data/fpinputf32rad_angles_fixed128
-- input@../data/fpinputf32rad_angles_fixed256
-- input@../data/fpinputf32rad_angles_fixed512
-- input@../data/fpinputf32rad_angles_fixed1024
-- input@../data/fpinputf32rad_angles_fixed1500
-- input@../data/fpinputf32rad_angles_fixed2000
-- input@../data/fpinputf32rad_angles_fixed2500
-- input@../data/fpinputf32rad_angles_fixed3000
-- input@../data/fpinputf32rad_angles_fixed3500
-- input@../data/fpinputf32rad_angles_fixed4000
-- input@../data/fpinputf32rad_angles_fixed4500
-- input@../data/fpinputf32rad_angles_fixed5000
-- input@../data/fpinputf32rad_detector_fixed100
-- input@../data/fpinputf32rad_detector_fixed445
-- input@../data/fpinputf32rad_detector_fixed790
-- input@../data/fpinputf32rad_detector_fixed1135
-- input@../data/fpinputf32rad_detector_fixed1480
-- input@../data/fpinputf32rad_detector_fixed1825
-- input@../data/fpinputf32rad_detector_fixed2170
-- input@../data/fpinputf32rad_detector_fixed2515
-- input@../data/fpinputf32rad_detector_fixed2860
-- input@../data/fpinputf32rad_detector_fixed3205
-- input@../data/fpinputf32rad_detector_fixed3550
-- input@../data/fpinputf32rad_detector_fixed3895
-- input@../data/fpinputf32rad_detector_fixed4240
-- input@../data/fpinputf32rad_detector_fixed4585
-- input@../data/fpinputf32rad_detector_fixed4930
-- input@../data/fpinputf32rad_detector_fixed5275
-- input@../data/fpinputf32rad_detector_fixed5620
-- input@../data/fpinputf32rad_detector_fixed5965
-- input@../data/fpinputf32rad_detector_fixed6310
-- input@../data/fpinputf32rad_detector_fixed6655

import "projection_lib"
open Projection

let main  [n](angles : []f32)
          (rhos : []f32)
          (image : *[n]f32): []f32 =
          let size = t32(f32.sqrt(r32(n)))
          let halfsize = size/2
          in forward_projection angles rhos halfsize image
