-- ==
-- input@../data/matrixinputf32rad128
-- input@../data/matrixinputf32rad256
import "line_lib"
open Lines
     -- function which computes the weight of pixels in grid_column for ray with entry/exit p
let calculate_weight(ent: point)
                    (ext: point)
                    (i: i32)
                    (N: i32) : [](f32,i32) =
     let Nhalf = N/2
     -- handle all lines as slope < 1 reverse the others
     let slope = (ext.2 - ent.2)/(ext.1 - ent.1)
     let reverse = slope > 1
     let k = if reverse then 1/slope else slope
     let gridentry = if reverse then (ent.2,ent.1) else ent
     --- calculate stuff
     let ymin = k*(r32(i) - gridentry.1) + gridentry.2 + r32(Nhalf)
     let yplus = k*(r32(i) + 1 - gridentry.1) + gridentry.2 + r32(Nhalf)
     let Ypixmin = t32(f32.floor(ymin))
     let Ypixplus = t32(f32.floor(yplus))
     let baselength = f32.sqrt(1+k ** 2.0f32)
     -- in [(t32(Ypixmin),baselength),(t32(Ypixplus),baselength)]
     let Ypixmax = i32.max Ypixmin Ypixplus
     let ydiff = yplus - ymin
     let yminfact = (r32(Ypixmax) - ymin)/ydiff
     let yplusfact = (yplus - r32(Ypixmax))/ydiff
     let lymin = yminfact*baselength
     let lyplus = yplusfact*baselength
     let iindex = i+Nhalf
     let pixmin = if reverse then iindex*N+Ypixmin else iindex+Ypixmin*N
     let pixplus = if reverse then iindex*N+Ypixplus else iindex+Ypixplus*N
     let min = if (pixmin >= 0 && pixmin < N ** 2) then
          (if Ypixmin == Ypixplus then (baselength,pixmin) else (lymin,pixmin))
          else (-1f32,-1i32)
     let plus = if (pixplus >= 0 && pixplus < N ** 2) then
          (if Ypixmin == Ypixplus then (-1f32,-1i32) else (lyplus,pixplus))
          else (-1f32,-1i32)
     in [min,plus]

-- assuming flat lines and gridsize even
let weights_doublepar    (angles: []f32)
                         (rays: []f32)
                         (gridsize: i32): [][](f32,i32) =
     let halfgridsize = gridsize/2
     let entryexitpoints =  convert2entryexit angles rays (r32(halfgridsize))
     in map(\(ent,ext) -> (flatten(map (\i ->
               calculate_weight ent ext i gridsize
          )((-halfgridsize)...(halfgridsize-1))))) entryexitpoints

let main  (angles: []f32)
          (rays: []f32)
          (gridsize: i32) : [][](f32,i32) =
     weights_doublepar angles rays gridsize
