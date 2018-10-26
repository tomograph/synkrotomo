import "sirtLIB"
open sirtLIB
import "intersect_lib"
open Intersections
import "futlib/array"

let unzip_d3 (xs: [][][](f32,i32)): ([][][]f32,[][][]i32) =
  xs |> map (map unzip)
     |> map unzip
     |> unzip

let unzip_d2 (xs: [][](f32,i32)): ([][]f32,[][]i32) =
 xs |> map unzip
    |> unzip


let main  (vect     : []f32)
          (angles   : []f32)
          (rays     : []f32)
          (gridsize : i32) =
  let halfsize = r32(gridsize)/2
  let entrypoints = convert2entry angles rays halfsize
  let sysmat = (map (\(p,sc) -> (lengths gridsize sc.1 sc.2 p)) entrypoints)
  let num_cols = gridsize*gridsize
  let num_rows = (length angles)*(length rays)
  in sparseMatMult sysmat vect num_cols num_rows
