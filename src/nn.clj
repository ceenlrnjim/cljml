(ns nn
  (:require [logregvec :as gd])
  (:require [incanter.core :as alg]))

(def INIT_EPSILON 0.12) ;

(defn sigmoid [x]
    ; 1 / (1 + exp(-x)
  (alg/div 1 (alg/plus 1 (alg/exp (alg/mult -1 x)))))

(defn sigmoid-gradient [x]
  ; note: element-wise
  (let [g (sigmoid x)]
      ; g .* (1 - g)
      (alg/mult g (alg/minus 1 g))))


(defn zeros
  [rows cols]
                   ; convert to all zeros by multiplying by 0
                                            ; get a collection of enough elements
  (alg/matrix (map (partial * 0) (range 0 (* rows cols))) cols))

(defn ones
  [rows cols]
                   ; convert to all ones by multiplying by 0 and adding one
                                            ; get a collection of enough elements
  ;(alg/matrix (map (comp inc (partial * 0)) (range 0 (* rows cols))) cols))
  (alg/plus 1 (zeros rows cols)))

(defn perturb-vector
  "Returns a vector of all zeros of the specified size except for the indicated index
  will have value e"
  [s ix e]
  (alg/matrix 
    (assoc
      (vec (map (partial * 0) (range 0 s)))
      ix 
      e)))

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

(defn unroll
  "Returns a vector of the content from all the matricies that are the values in the provided map"
  [mm]
  (alg/matrix (flatten (map alg/vectorize (vals mm)))))

(defn subvec-matrix
  "Returns a matrix of the specified dimension based on the appropriate number of
  elements from the specified row vector starting at index ix"
  [rv ix [rows cols]]
  (alg/matrix (alg/$ (range ix (+ ix (* rows cols))) rv) cols))

(defn els
  "returns the number elements in the specified dimension"
  [[rows cols]]
  (* rows cols))

(defn reroll
  "Returns a map of l to the matrix of parameters for layer l from the unrolled vector
  of parameter values and a sequence of the number of units in each layer from input (index 0)
  to output (index n-1)"
  [paramvec sizes]
  (loop [layerix 1
         lastendix 0
         result {}
         units sizes]
   (if (= (count units) 1) result
     (let [[sj sjplus1 & more] units
           mdim [sjplus1 (inc sj)]
           cells (* (mdim 0) (mdim 1))]
       (println "processing layer " layerix " with dimensions" mdim)
       (recur
         (inc layerix)
         (+ lastendix cells)
         (assoc result layerix (subvec-matrix paramvec lastendix mdim))
         (rest units))))))

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

;TODO: don't actually need z-vals, remove again
(defn forward-prop
  "Returns map containing :prediction (the predicted value of the network) and :activations (a map from layer to the activation values for that layer)
  based on inputs X (mxn) and map of theta values - keyed by layer, maps to theta value matrix.  X is NOT expected to
  have any X0 bias terms already added."
  [X thetas]
  (let [L (inc (count thetas))
        [setsize featurecount] (alg/dim X) ]
    (loop [unbiasedA X
           thetaix 1 
           result {:activations {1 X}}]
      (if (>= thetaix L)
        ; add the final result as the prediction and into the activations values
        (assoc-in (assoc result :prediction unbiasedA) [:activations L] unbiasedA)
        (do
          (println "Adding bias column to A (" (alg/dim unbiasedA)") of size [" setsize ",1]")
          (let [A (column-cat (ones setsize 1) unbiasedA)]
            (println "computing activation values for thetas" thetaix)
            (println "thetaix=" thetaix "L=" L "A=" (alg/dim A) "theta=" (alg/dim (thetas thetaix)))
              (let [nextZ (alg/mmult A (alg/trans (thetas thetaix)))
                    nextA (sigmoid nextZ)]
                (recur nextA 
                       (inc thetaix) 
                       (assoc-in result [:activations thetaix] A)))))))))
                              ; want to keep and return z and a values for use
                              ; later by back propagation
                              ;:activations (assoc (:activations result) thetaix ))A)
                             

(defn regularize
  "Supports both cost and gradient regularization"
  ; this is the cost use case
  ([lambda m thetamap]
   (regularize lambda m thetamap 2))
  ; this is the gradient use case
  ([lambda thetamap]
   (regularize lambda 1 thetamap 1))
  ; General case
  ([lambda lambda-divisor thetamap theta-power]
  ; not using reduce-kv as I want to modify each collection before reducing them
  (* (/ lambda (* 2 lambda-divisor))
    (reduce
      ; TODO: want to use incanter sum-of-squares but getting errors with number types for matrices
      #(+ %1 (Math/pow %2 theta-power))
      0
      ; flatten to one big list of theta values
      (flatten
        ; conver to a list of x1...xn terms
        (map
          ; take the value, drop the x0 term
          ; TODO: currently not removing the x0 term
          (comp alg/to-vect second) thetamap))))))

 
; TODO: determine if I want to unroll parameters - is it required for optimization libraries?
(defn cost
  "Given the specified parameters (map from l to params for layer l to l+1), training set inputs X
  (matrix of m rows and n columns) and yvec training set outputs (vector of size m rows by 1 column)
  and regularization parameter (lambda) regparam, what is the cost?  Note that Xo terms are not-expected to be included.
  Also note that Y could also be a vector where the output is multi-class classification"
  [thetas X Y regparam]
  (println "cost for (thetas,x,y)" thetas "|" X "|" Y)
  ; m = # training set examples, k = number of labels
  ; L = total number of layers
  {:pre [(= (first (alg/dim X)) (first (alg/dim Y)))]}
  (let [[m k] (alg/dim Y) ; useful parameters
        predictions (:prediction (forward-prop X thetas))]
    (println "All Predictions: " predictions "(dim " (alg/dim predictions) ") and all actuals" Y "(dim" (alg/dim Y)")")
    (let [subcosts (map #(example-cost %1 %2) predictions Y)
          costsize (count subcosts)
          regterm (regularize regparam m thetas)] ; forcing eval of the lazy sequence
      (+ regterm
          (* -1 
             (/ (reduce (fn [sum costi] (+ sum costi)) 0 subcosts)
              m))))))

(defn network-errors
  "Computes the current network error deltas.  Returns a map keyed by layer
  mapping to the error value."
  [{:keys [activations prediction]} thetamap Y]
  (let [L (count activations)]
    (loop [errorLast (alg/minus (activations L) Y)
           activations_current (activations (dec L))
           result { L errorLast }
           layerIndex (dec L)]
      (println "computing error for layer " layerIndex " of " L)
      (if (= 1 layerIndex) result
        (let [thetas (thetamap layerIndex)
              error (alg/mult 
                      (alg/mmult errorLast thetas) ; for L-1, mx1 * 1x3 for XNOR network
                      (alg/mult activations_current (alg/minus 1 activations_current)))]
          (recur error 
                 (activations (dec layerIndex)) ; this is a little awkward, move to let above?
                 (assoc result layerIndex error)
                 (dec layerIndex)))))))

(defn gradients
  "Uses back propagation to compute the gradients for specific network and inputs"
  [X thetamap Y regparam]
  ; TODO: ** this needs to return a function that takes the thetas and computes the gradient **
  ; Or it needs to be curried when used to a function of one parameter
  (let [[m n] (alg/dim X)
        fpr (forward-prop X thetamap)
        errs (network-errors fpr thetamap Y)]
    ; return a map from Layer to gradients for that layer
    ; TODO: is this the best format for returning?
    (reduce (fn [result [k v]]
              (assoc result k 
                     ; this is the calculation for the gradient value
                     ; for a specific layer
                     (alg/plus (alg/mult (/ 1 m) v) (regularize regparam thetamap))))
            {}
            errs)))

(defn estimate-gradients
  "computes a linear estimate for the gradients of the network described by thetamap
   and the cost function (as a function of the parameters J(theta)"
  ([thetamap costFn] (estimate-gradients thetamap costFn 0.0001))
  ([thetamap costFn epsilon]
   ; Note see ex4/computeNumericalGradient for basis of this implementation
   (let [thetas (unroll thetamap)
         [m _] (alg/dim thetas)]
     (loop [ix 0
            result (zeros m 1)]
       (if (= ix m) result
         (let [perturb (perturb-vector m ix epsilon)
               loss1 (costFn (alg/minus thetas perturb))
               loss2 (costFn (alg/plus thetas perturb))
               newval (/ (- loss2 loss1) (* 2 epsilon))]
           ;TODO: how to set the cell value to newval?
           (recur (inc ix) result)))))))
;
; gradient checking
(defn validate-gradients
  [bp-grads estimate-grads tolerance]
  )

(defn train-network
  "network layout is a map with keywords matching initialize-parameters. Returns a classifier function."
  [network-layout X Y regparam learning-rate max-iters]
  (let [gradient-fn (fn [thetamap] (gradients X thetamap Y regparam))
        cost-fn (fn [thetamap] (cost thetamap X Y regparam))
    ; currently using home-brew gradient descent
    ; TODO: map with multiple thetas will break the optimize function
    ; - probably need to unroll the parameters and this probably impacts the gradient function above
    opt-thetas (gd/optimize {:initial-thetas (initialize-parameters network-layout)
                          :learning-rate learning-rate
                          :cost-fn cost-fn
                          :gradient-fn gradient-fn
                          :max-iters max-iters})]
  (fn [x]
    (forward-prop x opt-thetas))))

; TODO can I use incanter/optimize to solve for the parameters?


