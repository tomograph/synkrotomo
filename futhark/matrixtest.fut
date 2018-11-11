-- ==
-- compiled input {
--   [[1,2,3],[4,5,6],[7,8,9]]
-- }
-- output {
--   [[2,3,4],[5,6,7],[8,9,10]]
-- }
let main  [n](image : [n][n]i32) : [n][n]i32 =
          --let image = unflatten n n im
          map(\row -> (map2 (+) (replicate n 1) row)) image
