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
-- input@../data/fpinputf32rad_detector_fixed10
-- input@../data/fpinputf32rad_detector_fixed15
-- input@../data/fpinputf32rad_detector_fixed20
-- input@../data/fpinputf32rad_detector_fixed25
-- input@../data/fpinputf32rad_detector_fixed30
-- input@../data/fpinputf32rad_detector_fixed35
-- input@../data/fpinputf32rad_detector_fixed40
-- input@../data/fpinputf32rad_detector_fixed45
-- input@../data/fpinputf32rad_detector_fixed50
-- input@../data/fpinputf32rad_detector_fixed55
-- input@../data/fpinputf32rad_detector_fixed60
-- input@../data/fpinputf32rad_detector_fixed65
-- input@../data/fpinputf32rad_detector_fixed70
-- input@../data/fpinputf32rad_detector_fixed75
-- input@../data/fpinputf32rad_detector_fixed80
-- input@../data/fpinputf32rad_detector_fixed85
-- input@../data/fpinputf32rad_detector_fixed90
-- input@../data/fpinputf32rad_detector_fixed95
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
-- input@../data/fpinputf32rad_scale_12
-- input@../data/fpinputf32rad_scale_16
-- input@../data/fpinputf32rad_scale_20
-- input@../data/fpinputf32rad_scale_24
-- input@../data/fpinputf32rad_scale_28
-- input@../data/fpinputf32rad_scale_32
-- input@../data/fpinputf32rad_scale_36
-- input@../data/fpinputf32rad_scale_40
-- input@../data/fpinputf32rad_scale_44
-- input@../data/fpinputf32rad_scale_48
-- input@../data/fpinputf32rad_scale_52
import "projection_lib"
open Projection

let main  [n](angles : []f32)
          (rhos : []f32)
          (image : *[n]f32): []f32 =
          let size = t32(f32.sqrt(r32(n)))
          let halfsize = size/2
          in forward_projection angles rhos halfsize image
