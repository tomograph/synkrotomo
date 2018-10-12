-- ==
-- compiled input {
--   [1.0f32, 2.0, 3.0, 4.0]
--   [1.0f32, 2.0, 3.0, 4.0]
-- }
-- output { [2.0f32, 4.0f32, 6.0f32, 8.0f32] }
import "sirtLIB"

let main [n]
         (vect1 : [n]f32) (vect2 : [n]f32) =
         sirtLIB.vecADD vect1 vect2
