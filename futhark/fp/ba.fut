-- ==
-- compiled input {
--   [1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
--   2
--   [1.0f32, 2.0, 3.0, 4.0]
--   [1.0f32, 2.0, 3.0, 4.0]
--   16
-- }
-- output { [30.0f32, 30.0f32] }

import "sirtLIB"
open sirtLIB
import "intersect_lib"
open Intersections

let unzip_d3 (xs: [][][](f32,i32)): ([][][]f32,[][][]i32) =
  xs |> map (map unzip)
     |> map unzip
     |> unzip

let unzip_d2 (xs: [][](f32,i32)): ([][]f32,[][]i32) =
 xs |> map unzip
    |> unzip


let main  (vect     : []f32)
          (num_cols : i32)
          (angles   : []f32)
          (rays     : []f32)
          (gridsize : i32) =
  let halfsize = r32(gridsize)/2
  let entrypoints = convert2entry angles rays halfsize
  let sysmat = unzip_d2(map (\(p,sc) -> (lengths gridsize sc.1 sc.2 p)) entrypoints)
  let test = map(\(i, _) -> trace i+0) sysmat
  in map (\matRow -> sirtVecVec matRow vect num_cols) sysmat
  -- let mat = unzip_d2(map (\(p,sc) -> (lengths gridsize sc.1 sc.2 p)) entrypoints)


  -- sirtLIB.sirtVecVec matRow vect num_cols
