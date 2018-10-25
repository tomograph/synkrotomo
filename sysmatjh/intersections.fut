
import "intersect_lib"
open Intersections

let unzip_d3 (xs: [][][](f32,i32)): ([][][]f32,[][][]i32) =
  xs |> map (map unzip)
     |> map unzip
     |> unzip

 let unzip_d2 (xs: [][](f32,i32)): ([][]f32,[][]i32) =
   xs |> map unzip
      |> unzip

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
