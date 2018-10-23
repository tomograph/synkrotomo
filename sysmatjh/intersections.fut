
import "intersect_lib"
open Intersections

let unzip_d3 (xs: [][][](f32,i32)): ([][][]f32,[][][]i32) =
  xs |> map (map unzip)
     |> map unzip
     |> unzip

 let unzip_d2 (xs: [][](f32,i32)): ([][]f32,[][]i32) =
   xs |> map unzip
      |> unzip

--entry main
--  (grid_size:  i32)
--  (delta:      f32)
--  (line_count: i32)
--  (scan_start: f32)
--  (scan_end:   f32)
--  (scan_step:  f32)
--  : ([][][]f32,[][][]i32) =
--  let numOfAngles = t32( (scan_end - scan_start) / scan_step )
--  let angles = map (\i -> scan_start + r32(i) * scan_step) (iota numOfAngles)
--  in unzip_d3 (map ( \angle ->
--      map (lengths grid_size angle delta) (-line_count...line_count-1)
--    ) angles)
  -- Compute lengths of lines in grid

  entry main
    -- list of the angles of the projections given in degrees
    (angles:  []f32)
    -- list of the rays per angle,
    --given in x-component for a coordinate system
    --overlaying the image with the center at the center of the image.
    (rays:    []f32)
    -- size of image assuming square
    (gridsize: i32)
    : ([][]f32,[][]i32) =
      let halfsize = r32(gridsize)/2
      let entrypoints = convert2entry angles rays halfsize
      in unzip_d2(map (\(p,sc) -> (lengths gridsize sc.1 sc.2 p)) entrypoints)
