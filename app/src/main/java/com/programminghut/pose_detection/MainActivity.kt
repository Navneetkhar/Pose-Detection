package com.programminghut.pose_detection

//import kotlinx.coroutines.flow.internal.NoOpContinuation.context
//import kotlin.coroutines.jvm.internal.CompletedContinuation.context
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.media.MediaPlayer
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.programminghut.pose_detection.ml.MovenetThunder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import com.programminghut.pose_detection.ml.Ownmodel as Model


class MainActivity : AppCompatActivity() {

    val paint = Paint()
    lateinit var imageProcessor: ImageProcessor
    lateinit var modelMovenet: MovenetThunder
    lateinit var model:Model
    lateinit var bitmap: Bitmap
    lateinit var imageView: ImageView
    lateinit var handler:Handler
    lateinit var handlerThread: HandlerThread
    lateinit var textureView: TextureView
    lateinit var cameraManager: CameraManager
    lateinit var button: Button
    lateinit var button2:Button
    var predict:Boolean=false
    var started:Boolean=false
    lateinit var timerTextView:TextView
    private var cameraDevice: CameraDevice? = null
    var poseValue:Int=0
    var count:Int=0
    val coroutineScope: CoroutineScope = CoroutineScope(Dispatchers.Main)
    var cameraNo=1
    val chair_angles=DoubleArray(6)
    val cobra_angles=DoubleArray(6)
    val tree_angles=DoubleArray(6)
    val dog_angles=DoubleArray(6)
    val warrior_angles=DoubleArray(6)
    lateinit var mediaPlayer:MediaPlayer
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        get_permissions()
        imageProcessor = ImageProcessor.Builder().add(ResizeOp(256, 256, ResizeOp.ResizeMethod.BILINEAR)).build()
        modelMovenet = MovenetThunder.newInstance(this)
        model = Model.newInstance(this)
        imageView = findViewById(R.id.imageView)
        textureView = findViewById(R.id.textureView)
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        chair_angles[0]=4.318142
        chair_angles[1]=4.510578
        chair_angles[2]=57.620475
        chair_angles[3]=57.303637
        chair_angles[4]=49.558220
        chair_angles[5]=49.629981

        cobra_angles[0]=4.924620
        cobra_angles[1]=8.974174
        cobra_angles[2]=27.329289
        cobra_angles[3]=24.849196
        cobra_angles[4]=1.006030
        cobra_angles[5]=1.759849

        tree_angles[0]=18.796275
        tree_angles[1]=23.507317
        tree_angles[2]=0.349697
        tree_angles[3]=27.560557
        tree_angles[4]=0.710464
        tree_angles[5]=72.173440

        dog_angles[0] = 6.311273
        dog_angles[1] = 8.257360
        dog_angles[2] = 32.492685
        dog_angles[3] = 30.950307
        dog_angles[4] = 0.587651
        dog_angles[5] = 0.974894

        warrior_angles[0] = 5.326370
        warrior_angles[1] = 5.622003
        warrior_angles[2] = 6.639268
        warrior_angles[3] = 27.621400
        warrior_angles[4] = 0.082060
        warrior_angles[5] = 5.928121

        paint.setColor(Color.YELLOW)
        button=findViewById(R.id.button)
        button.setOnClickListener {
            cameraDevice?.close()
            cameraNo = if (cameraNo == 1) 0 else 1
            openCamera()
        }
        button2=findViewById(R.id.button2)
        timerTextView=findViewById(R.id.textView)
        button2.setOnClickListener{


            timerTextView.visibility= View.VISIBLE
            predict=false
            count=0
            coroutineScope.launch {
                for (i in 5 downTo 0) {
                    // Update the timer TextView with the remaining time
                    timerTextView.text = "$i"
                    timerTextView.setTextColor(Color.WHITE)
                    // Delay for 1 second
                    delay(1000)
                }
                timerTextView.visibility=View.INVISIBLE
                predict=true
                started=false

            }
        }
        textureView.surfaceTextureListener = object:TextureView.SurfaceTextureListener{
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                openCamera()
            }

            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {

            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                bitmap = textureView.bitmap!!
                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                if(predict==true)
                {
                    var tensorImage = TensorImage(DataType.UINT8)
                    tensorImage.load(bitmap)
                    tensorImage = imageProcessor.process(tensorImage)

                    val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 256, 256, 3), DataType.UINT8)
                    inputFeature0.loadBuffer(tensorImage.buffer)
                    val outputs = modelMovenet.process(inputFeature0)
                    val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray
                    val h = 256
                    val w = 256
//                    for (i in 0..16){
//                        Log.d("${i}","${outputFeature0.get(i*3+1)*w}  ${outputFeature0.get(i*3)*h}")
//                    }
                    val canvas = Canvas(mutable)

                    var x = 0
                    val h_screen=bitmap.height
                    val w_screen=bitmap.width
    //                Log.d("output__", outputFeature0.size.toString())
                    while(x <= 49){
                        if(outputFeature0.get(x+2) > 0.1){
                            canvas.drawCircle(outputFeature0.get(x+1)*w_screen, outputFeature0.get(x)*h_screen, 15f, paint)
                        }
                        x+=3
                    }

                    val edgePaint=Paint()
                    edgePaint.setColor(Color.GREEN)
                    edgePaint.strokeWidth=7f
                    //edges making
                    //nose-left
                    if(outputFeature0.get(2)>0.1 && outputFeature0.get(2+3)>0.1){
                        canvas.drawLine(outputFeature0.get(1)*w_screen,outputFeature0.get(0)*h_screen,outputFeature0.get(1+3)*w_screen,outputFeature0.get(0+3)*h_screen,edgePaint)
                    }
                    //nose-right eye
                    if(outputFeature0.get(2)>0.1 && outputFeature0.get(2+6)>0.1){
                        canvas.drawLine(outputFeature0.get(1)*w_screen,outputFeature0.get(0)*h_screen,outputFeature0.get(1+6)*w_screen,outputFeature0.get(0+6)*h_screen,edgePaint)
                    }
                    //leftear-lefteye
                    if(outputFeature0.get(2+3)>0.1 && outputFeature0.get(2+9)>0.1){
                        canvas.drawLine(outputFeature0.get(1+3)*w_screen,outputFeature0.get(0+3)*h_screen,outputFeature0.get(1+9)*w_screen,outputFeature0.get(0+9)*h_screen,edgePaint)
                    }
                    //righteye-rightear
                    if(outputFeature0.get(2+6)>0.1 && outputFeature0.get(2+12)>0.1){
                        canvas.drawLine(outputFeature0.get(1+6)*w_screen,outputFeature0.get(0+6)*h_screen,outputFeature0.get(1+12)*w_screen,outputFeature0.get(0+12)*h_screen,edgePaint)
                    }
                    //left shoulder-nose
                    if(outputFeature0.get(2)>0.1 && outputFeature0.get(2+15)>0.1){
                        canvas.drawLine(outputFeature0.get(1)*w_screen,outputFeature0.get(0)*h_screen,outputFeature0.get(1+15)*w_screen,outputFeature0.get(0+15)*h_screen,edgePaint)
                    }
                    //right shoulder-nose
                    if(outputFeature0.get(2)>0.1 && outputFeature0.get(2+18)>0.1){
                        canvas.drawLine(outputFeature0.get(1)*w_screen,outputFeature0.get(0)*h_screen,outputFeature0.get(1+18)*w_screen,outputFeature0.get(0+18)*h_screen,edgePaint)
                    }
                    //right shoulder-left shoulder
                    if(outputFeature0.get(2+15)>0.1 && outputFeature0.get(2+18)>0.1){
                        canvas.drawLine(outputFeature0.get(1+15)*w_screen,outputFeature0.get(0+15)*h_screen,outputFeature0.get(1+18)*w_screen,outputFeature0.get(0+18)*h_screen,edgePaint)
                    }
                    //left shoulder-left elbow
                    if(outputFeature0.get(2+21)>0.1 && outputFeature0.get(2+15)>0.1){
                        canvas.drawLine(outputFeature0.get(1+21)*w_screen,outputFeature0.get(0+21)*h_screen,outputFeature0.get(1+15)*w_screen,outputFeature0.get(0+15)*h_screen,edgePaint)
                    }
                    //right shoulder-right elbow
                    if(outputFeature0.get(2+24)>0.1 && outputFeature0.get(2+18)>0.1){
                        canvas.drawLine(outputFeature0.get(1+24)*w_screen,outputFeature0.get(0+24)*h_screen,outputFeature0.get(1+18)*w_screen,outputFeature0.get(0+18)*h_screen,edgePaint)
                    }
                    //leftelbow-left wrist
                    if(outputFeature0.get(2+21)>0.1 && outputFeature0.get(2+27)>0.1){
                        canvas.drawLine(outputFeature0.get(1+21)*w_screen,outputFeature0.get(0+21)*h_screen,outputFeature0.get(1+27)*w_screen,outputFeature0.get(0+27)*h_screen,edgePaint)
                    }
                    //right elbow-right wrist
                    if(outputFeature0.get(2+24)>0.1 && outputFeature0.get(2+30)>0.1){
                        canvas.drawLine(outputFeature0.get(1+24)*w_screen,outputFeature0.get(0+24)*h_screen,outputFeature0.get(1+30)*w_screen,outputFeature0.get(0+30)*h_screen,edgePaint)
                    }
                    //left shoulder - left hip
                    if(outputFeature0.get(2+33)>0.1 && outputFeature0.get(2+15)>0.1){
                        canvas.drawLine(outputFeature0.get(1+33)*w_screen,outputFeature0.get(0+33)*h_screen,outputFeature0.get(1+15)*w_screen,outputFeature0.get(0+15)*h_screen,edgePaint)
                    }
                    //right shoulder- right hip
                    if(outputFeature0.get(2+36)>0.1 && outputFeature0.get(2+18)>0.1){
                        canvas.drawLine(outputFeature0.get(1+36)*w_screen,outputFeature0.get(0+36)*h_screen,outputFeature0.get(1+18)*w_screen,outputFeature0.get(0+18)*h_screen,edgePaint)
                    }
                    //left hip-right hip
                    if(outputFeature0.get(2+33)>0.1 && outputFeature0.get(2+36)>0.1){
                        canvas.drawLine(outputFeature0.get(1+33)*w_screen,outputFeature0.get(0+33)*h_screen,outputFeature0.get(1+36)*w_screen,outputFeature0.get(0+36)*h_screen,edgePaint)
                    }
                    //left hip-left knee
                    if(outputFeature0.get(2+33)>0.1 && outputFeature0.get(2+39)>0.1){
                        canvas.drawLine(outputFeature0.get(1+33)*w_screen,outputFeature0.get(0+33)*h_screen,outputFeature0.get(1+39)*w_screen,outputFeature0.get(0+39)*h_screen,edgePaint)
                    }
                    //right hip-right knee
                    if(outputFeature0.get(2+42)>0.1 && outputFeature0.get(2+36)>0.1){
                        canvas.drawLine(outputFeature0.get(1+42)*w_screen,outputFeature0.get(0+42)*h_screen,outputFeature0.get(1+36)*w_screen,outputFeature0.get(0+36)*h_screen,edgePaint)
                    }
                    //left knee-left ankle
                    if(outputFeature0.get(2+45)>0.1 && outputFeature0.get(2+39)>0.1){
                        canvas.drawLine(outputFeature0.get(1+45)*w_screen,outputFeature0.get(0+45)*h_screen,outputFeature0.get(1+39)*w_screen,outputFeature0.get(0+39)*h_screen,edgePaint)
                    }
                    //right knee- right ankle
                    if(outputFeature0.get(2+42)>0.1 && outputFeature0.get(2+48)>0.1){
                        canvas.drawLine(outputFeature0.get(1+42)*w_screen,outputFeature0.get(0+42)*h_screen,outputFeature0.get(1+48)*w_screen,outputFeature0.get(0+48)*h_screen,edgePaint)
                    }
                    var flag1=true
                    for(i in 0..16){
                        if(outputFeature0.get(i*3+2)<0.1){
                            flag1=false
                            break
                        }
                    }

                    if(flag1==true)
                    {
                        var maxPose=0
                        //centre of the pose(left hip and right hip)
                        val centreHipPoseX = ((outputFeature0.get(33)*w+outputFeature0.get(36)*w)*0.5)
                        val centreHipPoseY = ((outputFeature0.get(33+1)*h+outputFeature0.get(36+1)*h)*0.5)
                        //centre of pose(left shoulder and right shoulder)
                        val centreShoulderPoseX = ((outputFeature0.get(15)*w+outputFeature0.get(18)*w)*0.5)
                        val centreShoulderPoseY = ((outputFeature0.get(15+1)*h+outputFeature0.get(18+1)*h)*0.5)
                        //distance(body size - centrehip - centre shoulder)
                        val dist = sqrt((centreHipPoseX-centreShoulderPoseX).pow(2) +(centreHipPoseY-centreShoulderPoseY).pow(2) )
                        val coordinates= Array(17){FloatArray(2)}
                        for(i in 0..16){
                            coordinates[i][0]=(outputFeature0.get(i*3+0)*w-centreHipPoseX).toFloat()
                            coordinates[i][1]=(outputFeature0.get(i*3+1)*h-centreHipPoseY).toFloat()
                        }
                        //new centre of the body(hip centre)
                        val centreHipPoseX1=((coordinates[11][0]+coordinates[12][0])*0.5).toInt()
                        val centreHipPoseY1=((coordinates[11][1]+coordinates[12][1])*0.5).toInt()
                        //pose size calculations
                        val poseSize=dist*2.5
                        var pose_size=0.0f
                        for(i in 0..16){

                            pose_size= max(pose_size, sqrt((coordinates[i][0]-centreHipPoseX1).pow(2)+(coordinates[i][1]-centreHipPoseY1).pow(2)))
                        }
                        pose_size= max(pose_size,poseSize.toFloat())
                        for(i in 0..16){
                            coordinates[i][0]=coordinates[i][0]/pose_size
                            coordinates[i][1]=coordinates[i][1]/pose_size
                        }
                        var values=IntArray(5)
                        //

                        if(started==false)
                        {
                            val flattened = coordinates.flatMap { it.asList() }.toFloatArray()
                            val byteBuffer = ByteBuffer.allocateDirect(flattened.size * 4) // 4 bytes per float
                                .order(ByteOrder.nativeOrder())
                                .apply {
                                    asFloatBuffer().put(flattened)
                                    rewind()
                                }
                            val inputFeature1 = TensorBuffer.createFixedSize(intArrayOf(1, 34), DataType.FLOAT32)
                            inputFeature1.loadBuffer(byteBuffer)
                            val output = model.process(inputFeature1)
                            val outputFeature1 = output.outputFeature0AsTensorBuffer.floatArray
                            var predicted=0.0f
                            Log.d("output","${outputFeature1.get(0)}, ${outputFeature1.get(1)}, ${outputFeature1.get(2)}, ${outputFeature1.get(3)}, ${outputFeature1.get(4)}")
                            predicted= maxOf(outputFeature1.get(0),outputFeature1.get(1),outputFeature1.get(2),outputFeature1.get(3),outputFeature1.get(4))
                            if (predicted == outputFeature1.get(0)) {
                                poseValue = 0
                            } else if (predicted == outputFeature1.get(1)) {
                                poseValue = 1
                            } else if (predicted == outputFeature1.get(2)) {
                                poseValue = 2
                            } else if (predicted == outputFeature1.get(3)) {
                                poseValue = 3
                            } else if (predicted == outputFeature1.get(4)) {
                                poseValue = 4
                            }
                            values[poseValue]=values[poseValue]+1
                            count++
                        }
                        if(count==10 && started==false){
                            maxPose= maxOf(values[0],values[1],values[2],values[3],values[4])
                            if(maxPose==values[0])
                                maxPose=0
                            else if(maxPose==values[1])
                                maxPose=1
                            else if(maxPose==values[2])
                                maxPose=2
                            else if(maxPose==values[3])
                                maxPose=3
                            else if(maxPose==values[4])
                                maxPose=4
                            when (maxPose) {
                                0 -> {
                                    Toast.makeText(this@MainActivity, "Chair Pose Detected", Toast.LENGTH_LONG).show()
                                    mediaPlayer= MediaPlayer.create(this@MainActivity, R.raw.chair_pose)
                                    mediaPlayer.start()
                                }
                                1 -> {
                                    Toast.makeText(this@MainActivity, "Cobra Pose Detected", Toast.LENGTH_LONG).show()
                                    mediaPlayer= MediaPlayer.create(this@MainActivity, R.raw.cobra_pose)
                                    mediaPlayer.start()
                                }
                                2 -> {
                                    Toast.makeText(this@MainActivity, "Dog Pose Detected", Toast.LENGTH_LONG).show()
                                    mediaPlayer= MediaPlayer.create(this@MainActivity, R.raw.dog_pose)
                                    mediaPlayer.start()
                                }
                                3 -> {
                                    Toast.makeText(this@MainActivity, "Tree Pose Detected", Toast.LENGTH_LONG).show()
                                    mediaPlayer= MediaPlayer.create(this@MainActivity, R.raw.tree_pose)
                                    mediaPlayer.start()
                                }
                                4 -> {
                                    Toast.makeText(this@MainActivity, "Warrior Pose Detected", Toast.LENGTH_LONG).show()
                                    mediaPlayer= MediaPlayer.create(this@MainActivity, R.raw.warrior_pose)
                                    mediaPlayer.start()
                                }
                            }
//                            mediaPlayer.release()
                            started=true
                        }
                        if(started==true)
                        {
                            var angles = DoubleArray(6)
                            angles[0]=calculate_angle(coordinates[5][0],coordinates[5][1],coordinates[7][0],coordinates[7][1],coordinates[9][0],coordinates[9][1])
                            angles[1]=calculate_angle(coordinates[6][0],coordinates[6][1],coordinates[8][0],coordinates[8][1],coordinates[10][0],coordinates[10][1])
                            angles[2]=calculate_angle(coordinates[5][0],coordinates[5][1],coordinates[11][0],coordinates[11][1],coordinates[13][0],coordinates[13][1])
                            angles[3]=calculate_angle(coordinates[6][0],coordinates[6][1],coordinates[12][0],coordinates[12][1],coordinates[14][0],coordinates[14][1])
                            angles[4]=calculate_angle(coordinates[11][0],coordinates[11][1],coordinates[13][0],coordinates[13][1],coordinates[15][0],coordinates[15][1])
                            angles[5]=calculate_angle(coordinates[12][0],coordinates[12][1],coordinates[14][0],coordinates[14][1],coordinates[16][0],coordinates[16][1])
                            val flag = BooleanArray(6)
                            if(maxPose==0){
                                flag[0] = isValueInRange(chair_angles[0], angles[0])
                                flag[1] = isValueInRange(chair_angles[1], angles[1])
                                flag[2] = isValueInRange(chair_angles[2], angles[2])
                                flag[3] = isValueInRange(chair_angles[3], angles[3])
                                flag[4] = isValueInRange(chair_angles[4], angles[4])
                                flag[5] = isValueInRange(chair_angles[5],angles[5])
                                Log.d("0", angles[0].toString())
                                Log.d("1", angles[1].toString())
                                Log.d("2", angles[2].toString())
                                Log.d("3", angles[3].toString())
                                Log.d("4", angles[4].toString())
                                Log.d("5", angles[5].toString())
                            }
                            else if(maxPose==1){
                                flag[0] = isValueInRange(cobra_angles[0], angles[0])
                                flag[1] = isValueInRange(cobra_angles[1], angles[1])
                                flag[2] = isValueInRange(cobra_angles[2], angles[2])
                                flag[3] = isValueInRange(cobra_angles[3], angles[3])
                                flag[4] = isValueInRange(cobra_angles[4], angles[4])
                                flag[5] = isValueInRange(cobra_angles[5], angles[5])
                            }
                            else if(maxPose==2){
                                flag[0] = isValueInRange(dog_angles[0], angles[0])
                                flag[1] = isValueInRange(dog_angles[1], angles[1])
                                flag[2] = isValueInRange(dog_angles[2], angles[2])
                                flag[3] = isValueInRange(dog_angles[3], angles[3])
                                flag[4] = isValueInRange(dog_angles[4], angles[4])
                                flag[5] = isValueInRange(dog_angles[5], angles[5])
                            }
                            else if(maxPose==3){
                                flag[0] = isValueInRange(tree_angles[0], angles[0])
                                flag[1] = isValueInRange(tree_angles[1], angles[1])
                                flag[2]=true
                                flag[3]=true
//                                flag[2] = isValueInRange(tree_angles[2], angles[2])
//                                flag[3] = isValueInRange(tree_angles[3], angles[3])
                                flag[4] = isValueInRange(tree_angles[4], angles[4])
                                flag[5] = isValueInRange(tree_angles[5], angles[5])
                                Log.d("0", angles[0].toString())
                                Log.d("1", angles[1].toString())
                                Log.d("2", angles[2].toString())
                                Log.d("3", angles[3].toString())
                                Log.d("4", angles[4].toString())
                                Log.d("5", angles[5].toString())
                            }
                            else if (maxPose==4){
                                flag[0] = isValueInRange(warrior_angles[0], angles[0])
                                flag[1] = isValueInRange(warrior_angles[1], angles[1])
                                flag[2] = isValueInRange(warrior_angles[2], angles[2])
                                flag[3] = isValueInRange(warrior_angles[3], angles[3])
                                flag[4] = isValueInRange(warrior_angles[4], angles[4])
                                flag[5] = isValueInRange(warrior_angles[5], angles[5])
                            }
                            if(flag[0]==false){

                                if(!mediaPlayer.isPlaying){
                                    mediaPlayer= MediaPlayer.create(this@MainActivity, R.raw.left_elbow)
                                    mediaPlayer.start()
                                }
                            }
                            if(flag[1]==false){
                                if(!mediaPlayer.isPlaying){
                                    mediaPlayer= MediaPlayer.create(this@MainActivity, R.raw.right_elbow)
                                    mediaPlayer.start()
                                }
                            }
                            if(flag[2]==false){

                                if(!mediaPlayer.isPlaying){
                                    mediaPlayer= MediaPlayer.create(this@MainActivity, R.raw.left_hip)
                                    mediaPlayer.start()
                                }
                            }
                            if(flag[3]==false){

                                if(!mediaPlayer.isPlaying){
                                    mediaPlayer= MediaPlayer.create(this@MainActivity, R.raw.right_hip)
                                    mediaPlayer.start()
                                }
                            }
                            if(flag[4]==false){

                                if(!mediaPlayer.isPlaying){
                                    mediaPlayer= MediaPlayer.create(this@MainActivity, R.raw.left_knee)
                                    mediaPlayer.start()
                                }
                            }
                            if(flag[5]==false){

                                if(!mediaPlayer.isPlaying){
                                    mediaPlayer= MediaPlayer.create(this@MainActivity, R.raw.right_knee)
                                    mediaPlayer.start()
                                }
                            }
                        }
                    }

                }
                imageView.setImageBitmap(mutable)
            }
            fun calculate_angle(x1:Float,y1:Float,x2:Float,y2:Float,x3:Float,y3:Float): Double {
                val ab= sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
                val bc= sqrt((x3-x2).pow(2)+(y3-y2).pow(2))
                val ac= sqrt((x3-x1).pow(2)+(y3-y1).pow(2))
                val angleRad = acos((ab.pow(2) + ac.pow(2) - bc.pow(2)) / (2 * ab * ac))
                val angleDeg = Math.toDegrees(angleRad.toDouble())
                return angleDeg
            }
            fun isValueInRange(storedValue: Double, valueToCheck: Double): Boolean {
                val minDeviationValue = storedValue - 50
                val maxDeviationValue = storedValue + 50

                return valueToCheck in minDeviationValue..maxDeviationValue
            }
        }

    }

    override fun onDestroy() {
        super.onDestroy()
        modelMovenet.close()
        model.close()
        mediaPlayer.release()
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        cameraManager.openCamera(cameraManager.cameraIdList[cameraNo], object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) {
                cameraDevice = device
                val captureRequest = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                val surface = Surface(textureView.surfaceTexture)
                captureRequest.addTarget(surface)
                device.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        session.setRepeatingRequest(captureRequest.build(), null, null)
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e("Camera", "Failed to configure camera session.")
                    }
                }, handler)
            }

            override fun onDisconnected(device: CameraDevice) {
                cameraDevice?.close()
                cameraDevice = null
            }

            override fun onError(device: CameraDevice, error: Int) {
                cameraDevice?.close()
                cameraDevice = null
                Log.e("Camera", "Camera device error: $error")
            }
        }, handler)
    }



    fun get_permissions(){
        if(checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }
    override fun onRequestPermissionsResult(  requestCode: Int, permissions: Array<out String>, grantResults: IntArray  ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(grantResults[0] != PackageManager.PERMISSION_GRANTED) get_permissions()
    }
}