<? //https://youtu.be/HA-F6cZPvrg
function array_map_recursive($f, $data){ $result = []; if (!is_array($data)) return ($f($data)); foreach ($data as $k => $v) { $result[$k] = (is_array($v)) ? array_map_recursive($f, $v) : $f($v); } return $result;}
function subScalarVector($scalar, $data){ $result = [];foreach ($data as $key=>$value) {$result[$key] = $scalar - $value;}return $result;}
function subMatrixMatrix($data1, $data2){ $b = []; for ($i = 0; $i < count($data1); ++$i) { for ($j = 0; $j < count($data1[0]); ++$j) { $b[$i][$j] = $data1[$i][$j] - $data2[$i][$j]; }} return $b;}
function dotScalarRecursive($data, $scalar){ if (is_array($data)) {foreach ($data as $key => $value) { $data[$key] = dotScalarRecursive($value, $scalar); }return $data; } return $data * $scalar;}
function dotMatrixVector(array $data1, array $data2){ if (is_array($data1[0])) { foreach ($data1 as $key => $value) { $data1[$key] = dotMatrixVector($value, $data2); } return $data1; } return dotVectorVector($data1, $data2);}
function dotVectorVector(array $data1, array $data2){ if (count($data1) !== count($data2)) { die(sprintf("Vector size %d is different to vector size %d", count($data1), count($data2) ) ); } $sum = 0; foreach ($data1 as $key => $value) { $sum += $value * $data2[$key]; } return $sum;}
function dotMatrixMatrix(array $data1, array $data2){ $product = []; $size2 = count($data2[0]); foreach ($data1 as $i => $rowI) { $row = []; for ($j = 0; $j < $size2; $j++) { $sum = 0; foreach ($rowI as $key => $value1) { $sum += $value1 * $data2[$key][$j]; } $row[$j] = $sum; } $product[$i] = $row; } return $product;}
function MSE($y, $Y){ foreach ($y as $k=>$v) { $z[$k] = ($v - $Y[$k]) * ($v - $Y[$k]); } return (array_sum($z) / count($y)); }
function T($a){ $b = []; for ($i = 0; $i < count($a); ++$i) { for ($j = 0; $j < count($a[0]); ++$j) { $b[$j][$i] = $a[$i][$j]; }} return $b;}
function sigmoid($x) {return 1.0 / (1.0 + exp(-$x));}
function sigmoid_dx($x) {return ($x * (1.0 - $x));}

$train = [[[0, 0, 0], 0],
          [[0, 0, 1], 1],
          [[0, 1, 0], 0],
          [[0, 1, 1], 0],
          [[1, 0, 0], 1],
          [[1, 0, 1], 1],
          [[1, 1, 0], 0],
          [[1, 1, 1], 0]];
class PartyNN  {
	function __construct($learning_rate=0.1) {
		$this->weights_0_1 = [[0.1,0.2,0.5],[0.5,0.5,0.3]];
		$this->weights_1_2 = [[.5,.5]];
		$this->w[0] = [[.79,.44,.43],[.85,.43,.29]];
		$this->w[1] = [[.5,.52]];
		//$this->weights_0_1 = $this->w[0];
		//$this->weights_1_2 = $this->w[1];
		//$this->w[0]=$this->weights_0_1;
		//$this->w[1]=$this->weights_1_2;

		$this->learning_rate = $learning_rate;
	}
	function predict($inputs) {
		$inputs_1 = is_array($inputs[0]) ? dotMatrixMatrix($this->weights_0_1,$inputs) : dotMatrixVector($this->weights_0_1,$inputs);
		$outputs_1 = array_map_recursive('sigmoid',$inputs_1);
		$inputs_2 = is_array($outputs_1[0]) ? dotMatrixMatrix($this->weights_1_2,$outputs_1) : dotMatrixVector($this->weights_1_2,$outputs_1);
		return array_map_recursive('sigmoid',$inputs_2[0]);
	}
	function mypredict($inputs) {
													
		$level = 0;										// Входящие данные, веса текущего слоя, выходящие данные;
		$vals = $inputs;									// Вход - исходные данные 
		$outputs = [];
		do {											// для всех слоёв...
	
		$weight = $this->w[$level];								// Веса; количество = количеству нейронов в сл.слое (!);
		$outputs[$level] = [];									// вЫходные значения; 
		
		for ($j=0; $j < count($weight); $j++) {							// Для всех вЫходящих

			$dot_sum = 0;									// общий вес
			foreach ($vals as $i=>$a) {							// для каждого входного значения
				$dot_sum += $vals[$i] * $weight[$j][$i];
			}
			$outputs[$level][$j] = sigmoid($dot_sum);					// Посчитать общий вес для одного вых. значения (нейрона).
		}
		$vals = $outputs[$level];								// Вход - нейроны предыдущего слоя

		} while (++$level < count($this->w));
		
		return $outputs[$level-1];
	}
	function mytrain($inputs, $expected_predict) {

		if (!is_array($expected_predict)) $expected_predict = [$expected_predict];
													
		$level = 0;										// Входящие данные, веса текущего слоя, выходящие данные;
		$vals = $inputs;									// Вход - исходные данные 
		$outputs = [];
		do {											// для всех слоёв...
	
		$weight = $this->w[$level];								// Веса; количество = количеству нейронов в сл.слое (!);
		$outputs[$level] = [];									// вЫходные значения; 

		for ($j=0; $j < count($weight); $j++) {							// Для всех вЫходящих

			$dot_sum = 0;									// общий вес
			foreach ($vals as $i=>$a) {							// для каждого входного значения
				$dot_sum += $vals[$i] * $weight[$j][$i];
			}
			$outputs[$level][$j] = sigmoid($dot_sum);					// Посчитать общий вес для одного вых. значения (нейрона).
		}
		$vals = $outputs[$level];								// Вход - нейроны предыдущего слоя

		} while (++$level < count($this->w));


		$error = [];
		for ($j=0; $j < count($expected_predict); $j++) {					// Считаем ошибку от правильных входных данных ($expected_predict).
			$error[$j] = $outputs[$level-1][$j] - $expected_predict[$j];			// Ошибка для каждого нейрона			error - actual - expected
		}


		// Обратное распространение
		while ($level-- > 0) {

			$actuals = $outputs[$level];								// Текущие результаты, от прямого распространения
			$outputs_prev = $level > 0 ? $outputs[$level-1] : $inputs;				// Берем слой слева; для последнего уровня данные - это входящие данные;

			for ($j=0; $j < count($error); $j++) {							// Для всех вЫходных нейронов
				$wdelta = $error[$j] * sigmoid_dx($actuals[$j]);				// Дельта весов				weights_delta = error * (sigmoid(x)dx)
				//print "\n\n\nwd = $wdelta, error[$j] = {$error[$j]}\n";			// посчитали ошибку, дельту для нейрона

				for ($i=0;$i< count($this->w[$level][$j]);$i++) {				// для всех весов нейрона

						$this->w[$level][$j][$i] =					//					weight_1 = weight_1 - 
						$this->w[$level][$j][$i] - 
							$outputs_prev[$i] * $wdelta * $this->learning_rate;	//					output1 * weights_delta * learning_rate
						$new_error[$i] = $wdelta * $this->w[$level][$j][$i];		// Само распространение ошибки, назад
				}
			}
			$error = $new_error;
		}

		
		
		

	}
	function train($inputs, $expected_predict) {     
		$inputs_1 = dotMatrixVector($this->weights_0_1,$inputs);		$outputs_1 = array_map('sigmoid',$inputs_1);
		$inputs_2 = dotMatrixVector($this->weights_1_2,$outputs_1);		$outputs_2 = array_map('sigmoid',$inputs_2);

		$actual_predict = $outputs_2[0];

		$error_layer_2 = $actual_predict - $expected_predict;
	        $gradient_layer_2 = $actual_predict * (1 - $actual_predict);
	        $weights_delta_layer_2 = $error_layer_2 * $gradient_layer_2 ;
		$tmp2 = dotScalarRecursive($outputs_1, $weights_delta_layer_2 * $this->learning_rate);
		$this->weights_1_2 = [array_map( function($data1,$data2){return $data1 - $data2;}, $this->weights_1_2[0], $tmp2 )];

		$error_layer_1 = dotScalarRecursive($this->weights_1_2, $weights_delta_layer_2);
		$gradient_layer_1 = array_map( function($data1,$data2){return $data1*$data2;}, subScalarVector(1,$outputs_1), $outputs_1 );
		$weights_delta_layer_1 = [array_map( function($data1,$data2){return $data1 * $data2;}, $error_layer_1[0], $gradient_layer_1 )];
		foreach($inputs as $key=>$val){ $re1[$key][0] = $val; }//reshape
		$tt = T(dotMatrixMatrix($re1,$weights_delta_layer_1 ));
		$this->weights_0_1 = subMatrixMatrix($this->weights_0_1, dotScalarRecursive($tt,$this->learning_rate) );
	}
} // class

$epochs = 1000;
$learning_rate = 0.1;
$network = new PartyNN( $learning_rate );

//$network->mytrain([1.,1.,0],0);
//print "\n\n\n";
//print_R($network->w);
//die;

for ($e = 0; $e < $epochs; $e++) {
    $inputs_ = $correct_predictions = [];
    foreach ($train as $key=>$val) {
	$network->mytrain($val[0],$val[1]);
	$inputs_[] = $val[0];
	$correct_predictions[] = $val[1];
    }
    //$train_loss = MSE($network->mypredict(T($inputs_)), $correct_predictions);
    //printf("\r Progress: %02d,  Training loss: %05f ", ( 100.0 * $e/$epochs), $train_loss);
}
//print ("\n");
//print_r ($network->mypredict([1.,1.,0]));

print_R($network->w);

foreach ($train as $key=>$val) {
	printf("\n For input: [%s] the prediction is: [%0.16f], expected: %s ",implode(",",$val[0]), ($network->mypredict($val[0])[0]), ($val[1]==1?'True':'False'));
	//print("\n For input: [".implode(",",$val[0])."] the prediction is: [".($network->mypredict($val[0])[0])."], expected: ".($val[1]==1?'True':'False'));
}
//print_r ($network->w);
