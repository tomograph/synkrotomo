-- Unit tests of transpose in 3D
-- ==
-- compiled input {
--  [0.0f32,1.0f32,2.0f32,3.0f32,4.0f32,5.0f32,6.0f32,7.0f32]
-- 0
-- 2
-- }
-- output { [0.0f32,1.0f32,2.0f32,3.0f32,4.0f32,5.0f32,6.0f32,7.0f32] }
-- compiled input {
--  [0.0f32,1.0f32,2.0f32,3.0f32,4.0f32,5.0f32,6.0f32,7.0f32]
-- 1
-- 2
-- }
-- output { [0.0f32,2.0f32,1.0f32,3.0f32,4.0f32,6.0f32,5.0f32,7.0f32] }
-- compiled input {
--  [0.0f32,1.0f32,2.0f32,3.0f32,4.0f32,5.0f32,6.0f32,7.0f32]
-- 2
-- 2
-- }
-- output { [0.0f32,4.0f32,2.0f32,6.0f32,1.0f32,5.0f32,3.0f32,7.0f32] }

import "forwardprojection_cb"
open fplib

let main  (volume: []f32)
          (transpdir: i32)
          (N: i32): []f32 =

          if (N < 10000 && transpdir == 0)
                        then volume else if (N < 10000 && transpdir == 1)
                        then flatten_3d <| transpose_xy <| copy (unflatten_3d N N N volume)
                        else if (N < 10000 && transpdir == 2)
                        then flatten_3d <| transpose_xz <| copy (unflatten_3d N N N volume)
                        else (replicate (N*N*N) 1.0f32)
