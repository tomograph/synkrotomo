-- ==
-- compiled input {
--   [1.0f32, 2.0, 3.0, 4.0]
--   [1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
--   2
-- }
-- output { [30.0f32, 30.0f32] }
import "sirtLIB"

-- let main [n]
--          (vect1 : [n]f32) (vect2 : [n]f32) =
--          sirtLIB.vecADD vect1 vect2
let main [m] [q]
          (matRow : [m]f32)
          (vect : [q]f32)
          (num_cols : i32) =
  sirtLIB.sirtVecVec matRow vect num_cols
