import "mettes_lib"
open Mette
-- validate calculate_weight
-- let main  (x1: f32) (y1: f32) (x2: f32) (y2: f32) (i: i32) (N: i32)=
--      unzip3(calculate_weight (x1,y1) (x2,y2) i N)

let main(angles: []f32) (rays: []f32) (gridsize: i32) (vector: []f32): []f32 =
     forwardprojection angles rays gridsize vector

-- let unzip_d2 (xs: [][](i32,f32)): ([][]i32,[][]f32) =
--  xs |> map unzip
--      |> unzip
--
-- let main(angles: []f32) (rays: []f32) (gridsize: i32): ([][]i32,[][]f32) =
--      unzip_d2((compute_weights angles rays gridsize))
