(ns nn.test
    (:use nn)
    (:require [incanter.core :as alg])
    (:use clojure.test))


(defn- closeto?
  ([x y] (closeto? x y 0.001))
  ([x y tolerance] (< (Math/abs (- y x)) tolerance)))

(deftest test-sigmoid
  (is (= (sigmoid 0) 0.5))
  (is (< (sigmoid -10) 0.0001))
  (is (> (sigmoid 10) 0.9999))
  (let [m (alg/matrix [[-10 0 10] [-10 0 10] [-10 0 10]])
        sigm (sigmoid m)]
    (is (< (alg/$ 0 0 sigm) 0.0001))
    (is (= (alg/$ 0 1 sigm) 0.5))
    (is (> (alg/$ 0 2 sigm) 0.9999))))

(deftest test-sigmoid-gradient
  (let [m (alg/matrix [[1 2 3] [4 5 6]])
        sgm (sigmoid-gradient m)
        octave-answer (alg/matrix [[0.1966119 0.1049936 0.0451767][0.0176627 0.0066481 0.0024665]])]
    (is (every? (partial closeto? 0) (alg/vectorize (alg/minus octave-answer sgm))))))

(deftest test-drop-column
  (let [m (alg/matrix [[1 2 3][1 2 3][1 2 3]])]
    (is (= (drop-column m 0) (alg/matrix [[2 3][2 3][2 3]])))
    (is (= (drop-column m 1) (alg/matrix [[1 3][1 3][1 3]])))
    (is (= (drop-column m 2) (alg/matrix [[1 2][1 2][1 2]])))))

(deftest test-rand-theta
  (let [m (rand-theta 3 2)]
    (is (alg/matrix? m))
    (is (= (alg/dim m) [3 2]))
    (is (not (every? (partial = 0) m)))))

(deftest test-initialize-parameters
  (let [m (initialize-parameters {:innodes 5
                                  :outnodes 1
                                  :hidden-layers 2
                                  :hidden-units 10})]
    (is (= (count m) 3))
    (is (= (alg/dim (m 1)) [10 6]))
    (is (= (alg/dim (m 2)) [10 11]))
    (is (= (alg/dim (m 3)) [1 11]))))

(deftest test-column-cat
  ; test the multi-dimensional matrix case
  (is (= (alg/matrix [[1 2 3 4] [1 2 3 4] [1 2 3 4][1 2 3 4]])
        (column-cat (alg/matrix [[1 2][1 2][1 2][1 2]])
                    (alg/matrix [[3 4][3 4][3 4][3 4]]))))
  ; test two single dimensional vertical vectors
  (is (= (alg/matrix [[1 2][1 2][1 2][1 2]]) 
        (column-cat (alg/matrix [[1][1][1][1]]) (alg/matrix [[2][2][2][2]]))))
  ; test for the pain-in-the-ass matrix to vector conversion stuff
  (let [x (alg/matrix [[1.0 1.0]])
        y (alg/matrix [[1.0]])
        result (column-cat x y)]
    (is (=  (alg/matrix result [[1.0 1.0 1.0]])))
    (is (= (alg/dim result) [1 3])))
  ; test for more single dimension stuff
  (is (= (alg/matrix [[1 2] [1 2] [1 2]])
        (column-cat (alg/matrix [1 1 1]) (alg/matrix [[2][2][2]])))))

(comment
(deftest test-forward-prop
  ; using a XNOR logic "gate" to test since I can pick the theta values manually
  (let [X (alg/matrix [[0 0]])
        thetas {1 (alg/matrix [[-30 20 20][10 -20 -20]])
                2 (alg/matrix [[-10 20 20]])}
        prediction (forward-prop X thetas)]
    (is (every? #(closeto? 0 %) (alg/minus (alg/matrix [[1.0]]) prediction))))
  (let [X (alg/matrix [[0 0][0 1] [1 0][1 1]])
        thetas {1 (alg/matrix [[-30 20 20][10 -20 -20]])
                2 (alg/matrix [[-10 20 20]])}
        prediction (forward-prop X thetas)]
    (is (every? #(closeto? 0 %) (alg/minus (alg/matrix [[1.0][0.0][0.0][1.0]]) prediction)))))

; TODO: can't have a 0 in prediction or log goes to -infinity
(deftest test-example-cost
  ; answers come from octave
  (is (closeto? -0.030151 (example-cost (alg/matrix [[0.99 0.01 0.01]]) (alg/matrix [[1 0 0]]))))
  (is (closeto? -0.16252 (example-cost (alg/matrix [[0.85]]) (alg/matrix [[1]]))))
  (is (closeto? -2.2493 (example-cost (alg/matrix [[0.25 0.25 0.25 0.25]]) (alg/matrix [[0 0 1 0]]))))
  (is (closeto? 0.0 (example-cost 
                      (forward-prop
                        (alg/matrix [[0 0]])
                        ; xnor network
                        {1 (alg/matrix [[-30 20 20][10 -20 -20]])
                         2 (alg/matrix [[-10 20 20]])})
                      (alg/matrix [[1.0]])))))

(deftest test-cost
  (let [X (alg/matrix [[0 0][0 1] [1 0][1 1]])
        thetas {1 (alg/matrix [[-30 20 20][10 -20 -20]])
                2 (alg/matrix [[-10 20 20]])}
        xnorcost (cost thetas X (alg/matrix [[1.0][0.0][0.0][1.0]]) 0.0)]
    (is (closeto? 0.0 xnorcost))
    ))
 ) 

