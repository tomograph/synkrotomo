-- Unit tests of indexing into volume from coordinate
-- ==
-- compiled input {
--  -1.0f32
--  0
--  0.5f32
--  2i32
-- }
-- output { 0 }
-- compiled input {
--  -0.8f32
--  -1
--  -0.5f32
--  2i32
-- }
-- output { 6 }
-- compiled input {
--  -0.5f32
--  -1
--  0.5f32
--  4i32
-- }
-- output { 25 }

import "forwardprojection_cb"
open fplib

let main  (x: f32)
          (y: i32)
          (z: f32)
          (N: i32): i32 =
          pixel_index x y z N
