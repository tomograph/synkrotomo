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
  -- in handleMat (unzip_d2 sysmat)
  in sparseMatMult sysmat vect num_cols num_rows

  -- let sysmatworks = map(\(data, indexes) -> scatter (replicate num_cols 0) indexes data) sysmat


-- let (data, indexes) = unzip sysmat
  -- in map (\i -> map (\j -> scatter (replicate (gridsize*gridsize) 0) sysmat[1][i] sysmat[0][i] ) (iota (gridsize*gridsize)) ) (iota (length angles)*(length rays))
  -- in map (\i -> let r = replicate (gridsize*gridsize) 0 in map (\j -> if ) (iota r) ) (iota (length angles)*(length rays))


  -- in test = map(\i -> trace i+0) mw
  -- in map (\matRow -> sirtVecVec matRow vect num_cols) sysmat
  -- let mat = unzip_d2(map (\(p,sc) -> (lengths gridsize sc.1 sc.2 p)) entrypoints)


  -- sirtLIB.sirtVecVec matRow vect num_cols
--zgh600
--xmf768
