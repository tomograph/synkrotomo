-- ==
-- input@../data/bpinputf32rad64
-- input@../data/bpinputf32rad128
-- input@../data/bpinputf32rad256
-- input@../data/bpinputf32rad512
-- input@../data/bpinputf32rad1024
-- input@../data/bpinputf32rad1500
-- input@../data/bpinputf32rad2000
-- input@../data/bpinputf32rad2048
-- input@../data/bpinputf32rad2500
-- input@../data/bpinputf32rad3000
-- input@../data/bpinputf32rad3500
-- input@../data/bpinputf32rad4000
-- input@../data/bpinputf32rad4096
-- input@../data/bpinputf32rad_angles_fixed64
-- input@../data/bpinputf32rad_angles_fixed128
-- input@../data/bpinputf32rad_angles_fixed256
-- input@../data/bpinputf32rad_angles_fixed512
-- input@../data/bpinputf32rad_angles_fixed1024
-- input@../data/bpinputf32rad_angles_fixed1500
-- input@../data/bpinputf32rad_angles_fixed2000
-- input@../data/bpinputf32rad_angles_fixed2500
-- input@../data/bpinputf32rad_angles_fixed3000
-- input@../data/bpinputf32rad_angles_fixed3500
-- input@../data/bpinputf32rad_angles_fixed4000
-- input@../data/bpinputf32rad_angles_fixed4500
-- input@../data/bpinputf32rad_angles_fixed5000
-- input@../data/bpinputf32rad_detector_fixed100
-- input@../data/bpinputf32rad_detector_fixed445
-- input@../data/bpinputf32rad_detector_fixed790
-- input@../data/bpinputf32rad_detector_fixed1135
-- input@../data/bpinputf32rad_detector_fixed1480
-- input@../data/bpinputf32rad_detector_fixed1825
-- input@../data/bpinputf32rad_detector_fixed2170
-- input@../data/bpinputf32rad_detector_fixed2515
-- input@../data/bpinputf32rad_detector_fixed2860
-- input@../data/bpinputf32rad_detector_fixed3205
-- input@../data/bpinputf32rad_detector_fixed3550
-- input@../data/bpinputf32rad_detector_fixed3895
-- input@../data/bpinputf32rad_detector_fixed4240
-- input@../data/bpinputf32rad_detector_fixed4585
-- input@../data/bpinputf32rad_detector_fixed4930
-- input@../data/bpinputf32rad_detector_fixed5275
-- input@../data/bpinputf32rad_detector_fixed5620
-- input@../data/bpinputf32rad_detector_fixed5965
-- input@../data/bpinputf32rad_detector_fixed6310
-- input@../data/bpinputf32rad_detector_fixed6655

import "projection_lib"
open Projection

let main  [p](angles : []f32)
          (rhos : []f32)
          (size : i32)
          (projections: [p]f32): []f32 =
          let rhozero = unsafe rhos[0]
          let deltarho = unsafe rhos[1]-rhozero
          in back_projection angles rhozero deltarho size projections
