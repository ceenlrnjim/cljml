(ns nn
    (:require [incanter.core :as alg]))

(def INIT_EPSILON 0.12) ;

(defn sigmoid [x]
    ; 1 / (1 + exp(-x)
    (let [sigcalc #(/ 1 (+ 1 (alg/exp (* -1 %))))]
        (if (alg/matrix? x) (alg/matrix (alg/matrix-map sigcalc x)) (sigcalc x))))

(defn sigmoid-gradient [x]
  ; note: element-wise
  (let [g (sigmoid x)]
      ; g .* (1 - g)
      (alg/mult g (alg/minus 1 g))))

(defn ones
  [rows cols]
                   ; convert to all ones by multiplying by 0 and adding one
                                            ; get a collection of enough elements
  (alg/matrix (map (comp inc (partial * 0)) (range 0 (* rows cols))) cols))

; TODO: switch to incanter.core.conj-cols
(defn column-cat
  "Concatenates the specified matricies together, adding Y as additional columns to X"
  [x y]
  { :pre [(= (first (alg/dim x)) (first (alg/dim y)))] ; must have same number of rows
    :post [(= (alg/dim %) [(first (alg/dim x)) (+ (second (alg/dim x)) (second (alg/dim y)))])]  ; should be same rows, sum of columns
  }
    ;(println "Concatenating " (alg/dim x) "and " (alg/dim y))
    ;(let [[xrows xcols] (alg/dim x)
    ;      [yrows ycols] (alg/dim y)
    ;      result (alg/matrix (flatten (interleave x y)) (+ xcols ycols))]
    ;  (println "Result" (alg/dim result))
    ;  result)
  (let [[xrows xcols] (alg/dim x)
        [yrows ycols] (alg/dim y)]
    ; incanter switches my rows to columns and messes up concatenation
    (if (= xrows 1) 
      (alg/to-matrix (alg/conj-cols [x] [y]))
      (alg/to-matrix (alg/conj-cols x y)))))

(defn drop-column
  [m ix]
  (let [trans-m (alg/trans m)]
      (alg/trans
        (concat
            (take ix trans-m)
            (drop (inc ix) trans-m)))))

(defn rand-theta
  "Returns a matrix of size out by in+1 containing some randomly initialized weights"
  ([rows cols] (rand-theta rows cols INIT_EPSILON))
  ([rows cols epsilon]
  {:post [(= (alg/dim %) [rows cols])]}
  ; reshape into a matrix of the right dimensions - matrix with cols columns
  (alg/matrix
     ; replace each cell value with appropriate random number
     (map
       (fn [_] (- (* (rand) 2 INIT_EPSILON) INIT_EPSILON))
       ; generate a collection with the right number of cells
       (range 0 (* rows cols)))
     cols)))

; TODO: should this be a map or just a vector with appropriate index
(defn initialize-parameters
  "Returns a map keyed by layer number (1 indexed) that maps to the theta matrix for that maps to layer [l+1]"
  [{input-units :innodes output-units :outnodes hidden-layers :hidden-layers hidden-units :hidden-units}]
  ; create a list of node sizes in order [input hidden hidden .... output]
  (let [layersizes (loop [result [input-units]
                          h-layers-left hidden-layers]
                     (if (= h-layers-left 0) (conj result output-units)
                       (recur (conj result hidden-units) (dec h-layers-left))))
        ; create sequence of dimensions for the thetas between each layer
        ; [1 5 5 2] => [(1 5) (5 5) (5 2)]
        thetadims (loop [sizes layersizes
                         result []]
                    (if (empty? (rest sizes)) result
                      (let [[in out] (take 2 sizes)]
                        (recur (rest sizes) (conj result [in out])))))]
    ; built the map of layer to theta
    (reduce #(assoc %1 (inc (count %1)) %2)
      {}
      ; convert each dimension into a matrix of random values
      (map #(let [[in out] %] (rand-theta out (inc in))) thetadims))))

(defn example-cost
  "expects predict and actual as 1xK vectors or scalars"
  [predict actual]
  {:pre [(= (alg/dim predict) (alg/dim actual))]
   :post [(number? %)]}
          ; convert in case we're sent a scalar which will happen when K (number of classifications = 2, Y/N)
  (let [p (if (alg/matrix? predict) predict (alg/matrix [predict]))
        a (if (alg/matrix? actual) actual (alg/matrix[actual]))
        ptrans (alg/trans p)
        oneterm (alg/mmult a (alg/log ptrans))
        zeroterm (alg/mmult (alg/minus 1 a) (alg/log (alg/minus 1 ptrans)))]
      (+ oneterm zeroterm)))

(defn forward-prop
  "Returns the predicted value of the network based on inputs X (mxn) and
  map of theta values - keyed by layer, maps to theta value matrix.  X is NOT expected to
  have any X0 bias terms already added."
  [X thetas]
  (let [L (inc (count thetas))
        [setsize featurecount] (alg/dim X) ]
    (loop [unbiasedA X
           thetaix 1]
      (println "Adding bias column to A (" (alg/dim unbiasedA)") of size [" setsize ",1]")
      (let [A (column-cat (alg/matrix (ones setsize 1)) unbiasedA)]
        (println "computing activation values for thetas" thetaix)
        (println "thetaix=" thetaix "L=" L "A=" (alg/dim A) "theta=" (alg/dim (thetas thetaix)))
        (if (>= thetaix L)
          unbiasedA
            (let [nextZ (alg/mmult A (alg/trans (thetas thetaix)))
                  nextA (sigmoid nextZ)]
              (recur nextA (inc thetaix))))))))

(defn regularize
  [lambda m thetamap]
  ; not using reduce-kv as I want to modify each collection before reducing them
  (reduce
    #(+ %1 (Math/pow %2 2))
    0
    ; flatten to one big list of theta values
    (flatten
      ; conver to a list of x1...xn terms
      (map
        ; take the value, drop the x0 term
        #(rest (second %)) thetamap))))

 
; TODO: determine if I want to unroll parameters - is it required for optimization libraries?
(defn cost
  "Given the specified parameters (map from l to params for layer l to l+1), training set inputs X
  (matrix of m rows and n columns) and yvec training set outputs (vector of size m rows by 1 column)
  and regularization parameter (lambda) regparam, what is the cost?  Note that Xo terms are not-expected to be included.
  Also note that Y could also be a vector where the output is multi-class classification"
  [thetas X Y regparam]
  (println "cost for (thetas,x,y)" thetas "|" X "|" Y)
  ; TODO: probably want to keep z and a values for later
  ; m = # training set examples, k = number of labels
  ; L = total number of layers
  {:pre [(= (first (alg/dim X)) (first (alg/dim Y)))]}
  (let [[m k] (alg/dim Y) ; useful parameters
        predictions (forward-prop X thetas)]
    (println "All Predictions: " predictions "(dim " (alg/dim predictions) ") and all actuals" Y "(dim" (alg/dim Y)")")
    (let [subcosts (map #(example-cost %1 %2) predictions Y)
          costsize (count subcosts)
          regterm (regularize regparam m thetas)] ; forcing eval of the lazy sequence
      (+ regterm
          (* -1 
             (/ (reduce (fn [sum costi] (+ sum costi)) 0 subcosts)
              m))))))


; ------------------------------------------- Tested and working above here ---------------------------------
(comment
(defn count-layers
  "Returns the number of layers in the specified network"
  [nn]
  (+ 2 (:hidden-layers nn))


; network definition
(defn create-network
    "Returns a data structure that represents a neural network with the
     specified number of input and output units.  Each additional paramter is used as the number of units in
     a hidden layer (s2, s3, ...s(L-1))"
    [input-units output-units hidden-layers hidden-units]
    {:input-units input-units
     :output-units output-units
     :hidden-layers hidden-layers
     :hidden-units hidden-units
     ; Map from layer number (1 indexed) to the matrix of parameter values for that layer
     :parameters (initialize-parameters input-units output-units hidden-layers hidden-units)})


; backward prop
(defn backward-prop
  "Performs (Forward and) backward prop on a network for the specified training set to determine the
  gradients of the network.  returns a map from layer number to a seq of gradient functions with respect to each theta(l)"
  [nn X Y]
  ; gather gradient values for each theta
  (let [m (count X)]
      (map
        #(/ % m)
          (reduce
            (fn [gradients-vec [xi yi]]
              (let [fp (forward-prop nn xi); TODO: probably need matrix conversion here for xi
                    a-vec (:a fp)
                    z-vec (:z fp)
                    delta-last (alg/minus (last a-vec) yi)]
                  (loop [result gradients-vec
                         prev-delta delta-last
                         layer (dec (count-layers nn))]
                    (if (= 1 layer) result
                        (let [theta ((:parameters nn) (dec layer))
                              ; NOTE: this is elementwise multiplication, not matrix
                              delta (alg/mult
                                        (alg/mmult (alg/trans (drop-column theta 0) prev-delta)
                                                          ; TODO: dec?
                                        (sigmoid-gradient (z-vec layer))))]
                                                 ; 0 index structure
                                                                                                           ; TODO: dec?
                            (recur (assoc result (dec layer) (+ (result (dec layer)) (alg/mmult prev-delta (a-vec layer))))
                                   delta
                                   (dec layer)))))))
            (map (partial * 0) (range 1 (count-layers nn))) ; vector of zeros for each theta gradient
            (interleave X Y))))))

; training
(defn apply-gradients
  "Gradients is a sequence of the matrix of values for the derivative of the cost function with respect to the index of the
  matrix in the sequence (+1 for one indexed layers)"
  [gradients theta]
)

 
(defn gradient-descent
  "given a network (with its training sets, a cost function and a map of gradient functions, perform gradient descent
  and return the optimized set of theta values"
  [costfn gradients numiters alpha X Y, initialtheta]
  (let [m (count Y)]
    (loop [itercount 0
           theta initialtheta]
        (if (= itercount numiters) theta
            (recur (inc itercount)
                (alg/minus theta
                    (alg/mult alpha
                        (apply-gradients gradients theta))))))))

; prediction - re-use forward-prop?

) ; end comment

; gradient checking
(defn validate-gradients
  [backprop-gradient-map costFn]
  )

