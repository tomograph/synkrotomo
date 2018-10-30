module Mette = {
     type point  = ( f32, f32 )
     -- Degrees to radians converter
     let deg2rad(deg: f32): f32 = (f32.pi / 180f32) * deg
     -- function which computes the weights
     let calculate_weight(gridentry: point)
               (gridexit: point)
               (i: f32)
               (N: f32) : [](i32,f32) =
          let Nhalf = N/2f32
          let k = (gridexit.2 - gridentry.2)/(gridexit.1 - gridentry.1)
          let ymin = k*(i - gridentry.1) + gridentry.2 + Nhalf
          let yplus = k*(i + 1 - gridentry.1) + gridentry.2 + Nhalf
          let Ypixmin = f32.floor(ymin)
          let Ypixplus = f32.floor(yplus)
          let baselength = f32.sqrt(1+k ** 2.0f32)
          -- in [(t32(Ypixmin),baselength),(t32(Ypixplus),baselength)]
          let Ypixmax = f32.max Ypixmin Ypixplus
          let ydiff = yplus - ymin
          let yminfact = (Ypixmax - ymin)/ydiff
          let yplusfact = (yplus - Ypixmax)/ydiff
          let lymin = yminfact*baselength
          let lyplus = yplusfact*baselength
          let min = if 1 <= Ypixmin && Ypixmin <= N then
               (if Ypixmin == Ypixplus then (t32(Ypixmin),baselength) else (t32(Ypixmin),lymin))
               else (-1i32,-1f32)
          let plus = if 1 <= Ypixplus && Ypixplus <= N then
               (if Ypixmin == Ypixplus then (-1i32,-1f32) else (t32(Ypixplus),lyplus))
               else (-1i32,-1f32)
          in [min,plus]
}
