using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.UI;
using Emgu.Util;
using Emgu.CV.Structure;
using Emgu.CV.ML;
using Emgu.CV.Features2D;
using System.IO;
using System.Xml;
using System.Threading;
using Emgu.CV.CvEnum;
using System.Diagnostics;

namespace CNNClassifier
{
	public partial class Form1 : Form
	{
		public Form1()
		{
			InitializeComponent();
		}

		public static Image<Bgr, byte> ResizeUsingEmguCV(Image<Bgr, byte> original, int newWidth, int newHeight)
		{
			try
			{
				Image<Bgr, byte> image = original;
				Image<Bgr, byte> newImage = image.Resize(newWidth, newHeight, Inter.Cubic);
				return newImage;
			}
			catch
			{
				return null;
			}
		}

		public static Image<Gray, byte> ResizeUsingEmguCV(Image<Gray, byte> original, int newWidth, int newHeight)
		{
			try
			{
				Image<Gray, byte> image = original;
				Image<Gray, byte> newImage = image.Resize(newWidth, newHeight, Inter.Cubic);
				return newImage;
			}
			catch
			{
				return null;
			}
		}

		private void Form1_Load(object sender, EventArgs e)
		{
			CheckForIllegalCrossThreadCalls = false;

			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\0_正常\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\1_缆扭\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();
		}

		private void button1_Click(object sender, EventArgs e)
		{
			FolderBrowserDialog fDialog = new FolderBrowserDialog();
			fDialog.Description = "please select positve floder";

			if (fDialog.ShowDialog() == DialogResult.OK)
			{
				PSP.Text = fDialog.SelectedPath;
			}
		}

		private void button2_Click(object sender, EventArgs e)
		{
			FolderBrowserDialog fDialog = new FolderBrowserDialog();
			fDialog.Description = "please Negative positve floder";

			if (fDialog.ShowDialog() == DialogResult.OK)
			{
				NSP.Text = fDialog.SelectedPath;
			}
		}

		private void button3_Click(object sender, EventArgs e)
		{
			ThreadStart ts = new ThreadStart(trainSamples);
			Thread t = new Thread(ts);
			t.Start();
		}
		/*
         纬度信息计算方式:a = (winsize宽-block宽) / stride宽 +1,b = (winsize高-block高) / stride高 +1,c = block宽 / cell宽 ,d = block高/block高,纬度 = a*b*c*d*channel数，
		*/
		#region 初始化全局参数信息
		Matrix<float> HogFeatureData = new Matrix<float>(1200, 22500);
		Matrix<int> featureClasses = new Matrix<int>(1200, 1);//样本类别，1为正样本，-1为负样本

		SVM svm = new SVM();
		SVM svm0 = new SVM();
		SVM svm1 = new SVM();
		SVM svm2 = new SVM();
		SVM svm3 = new SVM();
		SVM svm4 = new SVM();
		SVM svm5 = new SVM();
		SVM svm6 = new SVM();
		SVM svm7 = new SVM();
		SVM svm8 = new SVM();
		SVM svm9 = new SVM();
		SVM svm10 = new SVM();
		SVM svm11 = new SVM();
		SVM svm12 = new SVM();
		SVM svm13 = new SVM();
		SVM svm14 = new SVM();
		SVM svm15 = new SVM();
		SVM svm16 = new SVM();
		SVM svm17 = new SVM();
		SVM svm18 = new SVM();
		SVM svm19 = new SVM();
		SVM svm20 = new SVM();
		//Predict的返回值
		float res = 0;

		int A, B, C, D, E, F, G = 0;//7种光纤类别

		int totalbar = 0;//总进度

		float TOTAL = 21;//总分类数目

		int total_count = 0;//图像数目
		int correct_count = 0;//正确图像
		double correct_recognition_rate = 0;//正确识别率

		Image<Bgr, byte> testimage = new Image<Bgr, byte>("test1.jpg");//测试图片1

		Image<Bgr, byte> testimage2 = new Image<Bgr, byte>("test2.jpg");//测试图片2

		//获取样本，并计算HOG特征
		HOGDescriptor hog = new HOGDescriptor(new Size(36, 36),
											  new Size(12, 12),
											  new Size(6, 6),
											  new Size(6, 6), 9);// 定义HOG描述子
		#endregion

		#region 训练图像
		private void trainSamples()//训练图像
		{
			DirectoryInfo diPos = new DirectoryInfo(PSP.Text);//正样本地址
			DirectoryInfo diNeg = new DirectoryInfo(NSP.Text);//副样本地址
			int posNum = diPos.GetFiles().Length;//图片数目
			int negNum = diNeg.GetFiles().Length;//
			FileInfo[] posFiles = diPos.GetFiles();//posFiles[0]={00001.jpg}
			FileInfo[] negFiles = diNeg.GetFiles();

			#region 正样本
			for (int fNum = 0; fNum < posNum; fNum++)
			{
				string filePath = diPos.FullName + "\\" + posFiles[fNum].Name;


				Image<Gray, byte> im = new Image<Gray, byte>(filePath);


				float[] fArr = hog.Compute(im);//计算HOG特征向量

				for (int i = 0; i < fArr.Length; i++)
				{
					HogFeatureData[fNum, i] = fArr[i];
				}

				featureClasses.Data[fNum, 0] = 1;

				#region 进度
				//PFN.Text = posFiles[fNum].Name;//文件中每一个图片的地址
				float tempF = (float.Parse((fNum + 1).ToString()) * 100 / float.Parse(posNum.ToString()));
				PPB.Value = (int)tempF;
				#endregion
			}

			#endregion

			#region 负样本
			for (int fNum = 0; fNum < negNum; fNum++)
			{
				string filePath = diNeg.FullName + "\\" + negFiles[fNum].Name;
				Image<Gray, byte> im = new Image<Gray, byte>(filePath);

				float[] fArr = hog.Compute(im);

				for (int i = 0; i < fArr.Length; i++)
				{
					HogFeatureData[fNum + posNum, i] = fArr[i];
				}

				featureClasses.Data[fNum + posNum, 0] = -1;

				#region 进度
				//NFN.Text = negFiles[fNum].Name;
				float tempF = (float.Parse((fNum + 1).ToString()) * 100 / float.Parse(negNum.ToString()));
				NPB.Value = (int)tempF;
				#endregion
			}

			#endregion


			#region 训练并保存训练结果

			svm = new SVM();
			svm.Type = SVM.SvmType.CSvc;
			svm.SetKernel(SVM.SvmKernelType.Linear);//线性
			svm.C = 1;
			svm.TermCriteria = new MCvTermCriteria(1000, 0.001);//1000次或者收敛达到0.001就跳出


			svm.Train(HogFeatureData, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, featureClasses);
			svm.Save("HogFeatures.xml");
			//MessageBox.Show("训练完毕，训练数据保存在：HogFeatures.xml！");
			#endregion

		}
		#endregion


		private void TestImage(Image<Bgr, byte> image, SVM _svm)//测试图像
		{


			Image<Gray, byte> _gimage = image.Convert<Gray, byte>();


			float[] TestArr = hog.Compute(_gimage);

			Matrix<float> samples = new Matrix<float>(1, 22500);

			for (int i = 0; i < TestArr.Length; i++)
			{
				samples[0, i] = TestArr[i];
			}

			res = _svm.Predict(samples);

			pictureBox1.Image = ResizeUsingEmguCV(image, 640, 480).ToBitmap();
		}

		private void button4_Click(object sender, EventArgs e)//关闭窗口
		{
			this.Close();
		}

		private void ZCVsLN()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\0_正常\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\1_缆扭\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm0 = svm;
			svm0.Save("HogFeatures_AB.xml");

			TestImage(testimage, svm0);



			if (res == 1)//正常
			{
				textBox1.Text = "正常图片";
				A++;
			}
			else if (res == -1)//揽扭
			{
				textBox1.Text = "缆扭图片";
				B++;
			}

			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			//显示A和B值
			textBox3.Text = A.ToString();
			textBox4.Text = B.ToString();
		}
		private void button5_Click(object sender, EventArgs e)//正常VS缆扭
		{
			ZCVsLN();
		}

		private void ZCVsLX()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\0_正常\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\2_露纤\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm1 = svm;
			svm1.Save("HogFeatures_AC.xml");

			TestImage(testimage, svm1);

			if (res == 1)//正常
			{
				textBox1.Text = "正常图片";
				A++;
			}
			else if (res == -1)//揽扭
			{
				textBox1.Text = "露纤图片";
				C++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox3.Text = A.ToString();
			textBox5.Text = C.ToString();
		}
		private void button6_Click(object sender, EventArgs e)//正常VS露纤
		{
			ZCVsLX();
		}

		private void ZCVsMS()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\0_正常\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\3_毛丝\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm2 = svm;
			svm2.Save("HogFeatures_AD.xml");

			TestImage(testimage, svm2);

			if (res == 1)//正常
			{
				textBox1.Text = "正常图片";
				A++;
			}
			else if (res == -1)//揽扭
			{
				textBox1.Text = "毛丝图片";
				D++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox3.Text = A.ToString();
			textBox6.Text = D.ToString();
		}
		private void button7_Click(object sender, EventArgs e)//正常VS毛丝
		{
			ZCVsMS();
		}

		private void ZCVsXJJ()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\0_正常\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\4_小节距\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm3 = svm;
			svm3.Save("HogFeatures_AE.xml");

			TestImage(testimage, svm3);

			if (res == 1)//正常
			{
				textBox1.Text = "正常图片";
				A++;
			}
			else if (res == -1)//揽扭
			{
				textBox1.Text = "小节距图片";
				E++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox3.Text = A.ToString();
			textBox7.Text = E.ToString();
		}
		private void button8_Click(object sender, EventArgs e)//正常VS小节距
		{
			ZCVsXJJ();

		}

		private void ZCVsYW()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\0_正常\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\5_异物\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm4 = svm;
			svm4.Save("HogFeatures_AF.xml");

			TestImage(testimage, svm4);

			if (res == 1)//正常
			{
				textBox1.Text = "正常图片";
				A++;
			}
			else if (res == -1)//揽扭
			{
				textBox1.Text = "异物图片";
				F++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox3.Text = A.ToString();
			textBox8.Text = F.ToString();
		}
		private void button9_Click(object sender, EventArgs e)//正常VS异物
		{
			ZCVsYW();
		}

		private void ZCVsYOUWU()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\0_正常\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\6_油污\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm5 = svm;
			svm5.Save("HogFeatures_AG.xml");

			TestImage(testimage, svm5);

			if (res == 1)//正常
			{
				textBox1.Text = "正常图片";
				A++;
			}
			else if (res == -1)//揽扭
			{
				textBox1.Text = "油污图片";
				G++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox3.Text = A.ToString();
			textBox9.Text = G.ToString();
		}
		private void button10_Click(object sender, EventArgs e)//正常VS油污
		{
			ZCVsYOUWU();

		}

		private void LNVsLX()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\1_缆扭\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\2_露纤\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm6 = svm;
			svm6.Save("HogFeatures_BC.xml");

			TestImage(testimage, svm6);

			if (res == 1)//缆扭
			{
				textBox1.Text = "缆扭图片";
				B++;
			}
			else if (res == -1)//露纤
			{
				textBox1.Text = "露纤图片";
				C++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox4.Text = B.ToString();
			textBox5.Text = C.ToString();
		}
		private void button11_Click(object sender, EventArgs e)//缆扭VS露纤
		{
			LNVsLX();

		}

		private void LNVsMS()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\1_缆扭\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\3_毛丝\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm7 = svm;
			svm7.Save("HogFeatures_BD.xml");

			TestImage(testimage, svm7);

			if (res == 1)//缆扭
			{
				textBox1.Text = "缆扭图片";
				B++;
			}
			else if (res == -1)//毛丝
			{
				textBox1.Text = "毛丝图片";
				D++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox4.Text = B.ToString();
			textBox6.Text = D.ToString();
		}
		private void button12_Click(object sender, EventArgs e)//缆扭VS毛丝
		{
			LNVsMS();

		}

		private void LNVsXJJ()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\1_缆扭\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\4_小节距\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm8 = svm;
			svm8.Save("HogFeatures_BE.xml");

			TestImage(testimage, svm8);

			if (res == 1)//缆扭
			{
				textBox1.Text = "缆扭图片";
				B++;
			}
			else if (res == -1)//小节距
			{
				textBox1.Text = "小节距图片";
				E++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox4.Text = B.ToString();
			textBox7.Text = E.ToString();
		}
		private void button13_Click(object sender, EventArgs e)//缆扭VS小节距
		{
			LNVsXJJ();

		}

		private void LNVsYW()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\1_缆扭\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\5_异物\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm9 = svm;
			svm9.Save("HogFeatures_BF.xml");

			TestImage(testimage, svm9);

			if (res == 1)//缆扭
			{
				textBox1.Text = "缆扭图片";
				B++;
			}
			else if (res == -1)//异物
			{
				textBox1.Text = "异物图片";
				F++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox4.Text = B.ToString();
			textBox8.Text = F.ToString();
		}
		private void button14_Click(object sender, EventArgs e)//缆扭VS异物
		{
			LNVsYW();

		}

		private void LNVsYOUWU()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\1_缆扭\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\6_油污\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm10 = svm;
			svm10.Save("HogFeatures_BG.xml");

			TestImage(testimage, svm10);

			if (res == 1)//缆扭
			{
				textBox1.Text = "缆扭图片";
				B++;
			}
			else if (res == -1)//油污
			{
				textBox1.Text = "油污图片";
				G++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox4.Text = B.ToString();
			textBox9.Text = G.ToString();
		}
		private void button15_Click(object sender, EventArgs e)//缆扭VS油污
		{
			LNVsYOUWU();

		}

		private void LXVsMS()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\2_露纤\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\3_毛丝\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm11 = svm;
			svm11.Save("HogFeatures_CD.xml");

			TestImage(testimage, svm11);

			if (res == 1)//露丝
			{
				textBox1.Text = "露纤图片";
				C++;
			}
			else if (res == -1)//毛丝
			{
				textBox1.Text = "毛丝图片";
				D++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox5.Text = C.ToString();
			textBox6.Text = D.ToString();
		}
		private void button16_Click(object sender, EventArgs e)//露纤VS毛丝
		{
			LXVsMS();

		}

		private void LXVsXJJ()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\2_露纤\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\4_小节距\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm12 = svm;
			svm12.Save("HogFeatures_CE.xml");

			TestImage(testimage, svm12);

			if (res == 1)//露纤
			{
				textBox1.Text = "露纤图片";
				C++;
			}
			else if (res == -1)//小节距
			{
				textBox1.Text = "小节距图片";
				E++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox5.Text = C.ToString();
			textBox7.Text = E.ToString();
		}
		private void button17_Click(object sender, EventArgs e)//露纤VS小节距
		{
			LXVsXJJ();

		}

		private void LXVsYW()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\2_露纤\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\5_异物\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm13 = svm;
			svm13.Save("HogFeatures_CF.xml");

			TestImage(testimage, svm13);

			if (res == 1)//露纤
			{
				textBox1.Text = "露纤图片";
				C++;
			}
			else if (res == -1)//异物
			{
				textBox1.Text = "异物图片";
				F++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox5.Text = C.ToString();
			textBox8.Text = F.ToString();
		}
		private void button18_Click(object sender, EventArgs e)//露纤VS异物
		{
			LXVsYW();

		}

		private void LXVsYOUWU()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\2_露纤\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\6_油污\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm14 = svm;
			svm14.Save("HogFeatures_CG.xml");

			TestImage(testimage, svm14);

			if (res == 1)//露纤
			{
				textBox1.Text = "露纤图片";
				C++;
			}
			else if (res == -1)//油污
			{
				textBox1.Text = "油污图片";
				G++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox5.Text = C.ToString();
			textBox9.Text = G.ToString();
		}
		private void button19_Click(object sender, EventArgs e)//露纤VS油污
		{
			LXVsYOUWU();

		}

		private void MSVsXJJ()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\3_毛丝\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\4_小节距\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm15 = svm;
			svm15.Save("HogFeatures_DE.xml");

			TestImage(testimage, svm15);

			if (res == 1)//毛丝
			{
				textBox1.Text = "毛丝图片";
				D++;
			}
			else if (res == -1)//小节距
			{
				textBox1.Text = "小节距图片";
				E++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox6.Text = D.ToString();
			textBox7.Text = E.ToString();
		}
		private void button20_Click(object sender, EventArgs e)//毛丝VS小节距
		{
			MSVsXJJ();

		}

		private void MSVsYW()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\3_毛丝\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\5_异物\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm16 = svm;
			svm16.Save("HogFeatures_DF.xml");

			TestImage(testimage, svm16);

			if (res == 1)//毛丝
			{
				textBox1.Text = "毛丝图片";
				D++;
			}
			else if (res == -1)//异物
			{
				textBox1.Text = "异物图片";
				F++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox6.Text = D.ToString();
			textBox8.Text = F.ToString();
		}
		private void button21_Click(object sender, EventArgs e)//毛丝VS异物
		{
			MSVsYW();

		}

		private void MSVsYOUWU()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\3_毛丝\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\6_油污\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm17 = svm;
			svm17.Save("HogFeatures_DG.xml");

			TestImage(testimage, svm17);

			if (res == 1)//毛丝
			{
				textBox1.Text = "毛丝图片";
				D++;
			}
			else if (res == -1)//油污
			{
				textBox1.Text = "油污图片";
				G++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox6.Text = D.ToString();
			textBox9.Text = G.ToString();
		}
		private void button22_Click(object sender, EventArgs e)//毛丝VS油污
		{
			MSVsYOUWU();

		}

		private void XJJVsYW()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\4_小节距\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\5_异物\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm18 = svm;
			svm18.Save("HogFeatures_EF.xml");

			TestImage(testimage, svm18);

			if (res == 1)//小节距
			{
				textBox1.Text = "小节距图片";
				E++;
			}
			else if (res == -1)//异物
			{
				textBox1.Text = "异物图片";
				F++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox7.Text = E.ToString();
			textBox8.Text = F.ToString();
		}
		private void button23_Click(object sender, EventArgs e)//小节距VS异物
		{
			XJJVsYW();

		}

		private void XJJVsYOUWU()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\4_小节距\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\6_油污\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm19 = svm;
			svm19.Save("HogFeatures_EG.xml");

			TestImage(testimage, svm19);

			if (res == 1)//小节距
			{
				textBox1.Text = "小节距图片";
				E++;
			}
			else if (res == -1)//油污
			{
				textBox1.Text = "油污图片";
				G++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox7.Text = E.ToString();
			textBox9.Text = G.ToString();
		}
		private void button24_Click(object sender, EventArgs e)//小节距VS油污
		{
			XJJVsYOUWU();

		}

		private void YWVsYOUWU()
		{
			DirectoryInfo di_P = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\5_异物\\test");
			DirectoryInfo di_N = new DirectoryInfo("C:\\Users\\yunti\\Desktop\\光圈数据集\\GuangQuan\\Train\\6_油污\\test");

			PSP.Text = di_P.FullName.ToString();
			NSP.Text = di_N.FullName.ToString();

			trainSamples();

			svm20 = svm;
			svm20.Save("HogFeatures_FG.xml");

			TestImage(testimage, svm20);

			if (res == 1)//异物
			{
				textBox1.Text = "异物图片";
				F++;
			}
			else if (res == -1)//油污
			{
				textBox1.Text = "油污图片";
				G++;
			}
			//清除之前的进度
			PPB.Value = 0;
			NPB.Value = 0;
			//总进度
			++totalbar;
			float temp = (float.Parse(totalbar.ToString()) * 100 / float.Parse(TOTAL.ToString()));
			progressBar1.Value = (int)temp;

			textBox8.Text = F.ToString();
			textBox9.Text = G.ToString();
		}
		private void button25_Click(object sender, EventArgs e)//异物VS油污
		{
			YWVsYOUWU();

		}

		private void button26_Click(object sender, EventArgs e)//投票
		{
			int[] a = { A, B, C, D, E, F, G };
			int MAX = 0;
			for (int i = 0; i < 7; i++)
			{
				if (a[i] > MAX)
				{
					MAX = a[i];
				}
			}
			if (MAX == A)
			{
				textBox2.Text = "正常图片";
			}
			else if (MAX == B)
			{
				textBox2.Text = "缆扭图片";
			}
			else if (MAX == C)
			{
				textBox2.Text = "露纤图片";
			}
			else if (MAX == D)
			{
				textBox2.Text = "毛丝图片";
			}
			else if (MAX == E)
			{
				textBox2.Text = "小节距图片";
			}
			else if (MAX == F)
			{
				textBox2.Text = "异物图片";
			}
			else if (MAX == G)
			{
				textBox2.Text = "油污图片";
			}
		}

		private void button27_Click(object sender, EventArgs e)//一键训练
		{
			totalbar = 0;//一键训练前先清所有进度条以及保存的数据
			progressBar1.Value = 0;
			PPB.Value = 0;
			NPB.Value = 0;
			A = 0;
			B = 0;
			C = 0;
			D = 0;
			E = 0;
			F = 0;
			G = 0;
			button5_Click(sender, e);
			button6_Click(sender, e);
			button7_Click(sender, e);
			button8_Click(sender, e);
			button9_Click(sender, e);
			button10_Click(sender, e);
			button11_Click(sender, e);
			button12_Click(sender, e);
			button13_Click(sender, e);
			button14_Click(sender, e);
			button15_Click(sender, e);
			button16_Click(sender, e);
			button17_Click(sender, e);
			button18_Click(sender, e);
			button19_Click(sender, e);
			button20_Click(sender, e);
			button21_Click(sender, e);
			button22_Click(sender, e);
			button23_Click(sender, e);
			button24_Click(sender, e);
			button25_Click(sender, e);
			MessageBox.Show("训练完毕，训练数据保存在：HogFeatures.xml！");
		}

		private void button28_Click(object sender, EventArgs e)//清除所有数据
		{
			//当出现投票结果出现相同的情况时，此时对相同结果再进行训练即可得出正确的结果
			//训练之前清除所有结果
			totalbar = 0;
			progressBar1.Value = 0;
			PPB.Value = 0;
			NPB.Value = 0;
			A = 0;
			B = 0;
			C = 0;
			D = 0;
			E = 0;
			F = 0;
			G = 0;
			textBox1.Text = null;
			textBox2.Text = null;
			textBox3.Text = null;
			textBox4.Text = null;
			textBox5.Text = null;
			textBox6.Text = null;
			textBox7.Text = null;
			textBox8.Text = null;
			textBox9.Text = null;

		}


		#region 利用XML初始化SVM，对传入的图像进行测试
		private void MultipleImageTest(Image<Bgr, byte> _mimage)
		{
			//AB
			#region
			TestImage(_mimage, svm0);

			if (res == 1)
			{
				textBox1.Text = "正常图片";
				A++;
			}
			else if (res == -1)
			{
				textBox1.Text = "缆扭图片";
				B++;
			}
			textBox3.Text = A.ToString();
			textBox4.Text = B.ToString();
			#endregion

			//AC
			#region
			TestImage(_mimage, svm1);

			if (res == 1)
			{
				textBox1.Text = "正常图片";
				A++;
			}
			else if (res == -1)
			{
				textBox1.Text = "露纤图片";
				C++;
			}
			textBox3.Text = A.ToString();
			textBox5.Text = C.ToString();
			#endregion

			//AD
			#region
			TestImage(_mimage, svm2);

			if (res == 1)
			{
				textBox1.Text = "正常图片";
				A++;
			}
			else if (res == -1)
			{
				textBox1.Text = "毛丝图片";
				D++;
			}
			textBox3.Text = A.ToString();
			textBox6.Text = D.ToString();
			#endregion

			//AE
			#region
			TestImage(_mimage, svm3);

			if (res == 1)
			{
				textBox1.Text = "正常图片";
				A++;
			}
			else if (res == -1)
			{
				textBox1.Text = "小节距图片";
				E++;
			}
			textBox3.Text = A.ToString();
			textBox7.Text = E.ToString();
			#endregion

			//AF
			#region
			TestImage(_mimage, svm4);

			if (res == 1)
			{
				textBox1.Text = "正常图片";
				A++;
			}
			else if (res == -1)
			{
				textBox1.Text = "异物图片";
				F++;
			}
			textBox3.Text = A.ToString();
			textBox8.Text = F.ToString();
			#endregion

			//AG
			#region
			TestImage(_mimage, svm5);

			if (res == 1)
			{
				textBox1.Text = "正常图片";
				A++;
			}
			else if (res == -1)
			{
				textBox1.Text = "油污图片";
				G++;
			}
			textBox3.Text = A.ToString();
			textBox9.Text = G.ToString();
			#endregion

			//BC
			#region
			TestImage(_mimage, svm6);

			if (res == 1)
			{
				textBox1.Text = "缆扭图片";
				B++;
			}
			else if (res == -1)
			{
				textBox1.Text = "露纤图片";
				C++;
			}
			textBox4.Text = B.ToString();
			textBox5.Text = C.ToString();
			#endregion

			//BD
			#region
			TestImage(_mimage, svm7);

			if (res == 1)
			{
				textBox1.Text = "缆扭图片";
				B++;
			}
			else if (res == -1)
			{
				textBox1.Text = "毛丝图片";
				D++;
			}
			textBox4.Text = B.ToString();
			textBox6.Text = D.ToString();
			#endregion

			//BE
			#region
			TestImage(_mimage, svm8);

			if (res == 1)
			{
				textBox1.Text = "缆扭图片";
				B++;
			}
			else if (res == -1)
			{
				textBox1.Text = "小节距图片";
				E++;
			}
			textBox4.Text = B.ToString();
			textBox7.Text = E.ToString();
			#endregion

			//BF
			#region
			TestImage(_mimage, svm9);

			if (res == 1)
			{
				textBox1.Text = "缆扭图片";
				B++;
			}
			else if (res == -1)
			{
				textBox1.Text = "异物图片";
				F++;
			}
			textBox4.Text = B.ToString();
			textBox8.Text = F.ToString();
			#endregion

			//BG
			#region
			TestImage(_mimage, svm10);

			if (res == 1)
			{
				textBox1.Text = "缆扭图片";
				B++;
			}
			else if (res == -1)
			{
				textBox1.Text = "油污图片";
				G++;
			}
			textBox4.Text = B.ToString();
			textBox9.Text = G.ToString();
			#endregion

			//CD
			#region
			TestImage(_mimage, svm11);

			if (res == 1)
			{
				textBox1.Text = "露纤图片";
				C++;
			}
			else if (res == -1)
			{
				textBox1.Text = "毛丝图片";
				D++;
			}
			textBox5.Text = C.ToString();
			textBox6.Text = D.ToString();
			#endregion

			//CE
			#region
			TestImage(_mimage, svm12);

			if (res == 1)
			{
				textBox1.Text = "露纤图片";
				C++;
			}
			else if (res == -1)
			{
				textBox1.Text = "小节距图片";
				E++;
			}
			textBox5.Text = C.ToString();
			textBox7.Text = E.ToString();
			#endregion

			//CF
			#region
			TestImage(_mimage, svm13);

			if (res == 1)
			{
				textBox1.Text = "露纤图片";
				C++;
			}
			else if (res == -1)
			{
				textBox1.Text = "异物图片";
				F++;
			}
			textBox5.Text = C.ToString();
			textBox8.Text = F.ToString();
			#endregion

			//CG
			#region
			TestImage(_mimage, svm14);

			if (res == 1)
			{
				textBox1.Text = "露纤图片";
				C++;
			}
			else if (res == -1)
			{
				textBox1.Text = "油污图片";
				G++;
			}
			textBox5.Text = C.ToString();
			textBox9.Text = G.ToString();
			#endregion

			//DE
			#region
			TestImage(_mimage, svm15);

			if (res == 1)
			{
				textBox1.Text = "毛丝图片";
				D++;
			}
			else if (res == -1)
			{
				textBox1.Text = "小节距图片";
				E++;
			}
			textBox6.Text = D.ToString();
			textBox7.Text = E.ToString();
			#endregion

			//DF
			#region
			TestImage(_mimage, svm16);

			if (res == 1)
			{
				textBox1.Text = "毛丝图片";
				D++;
			}
			else if (res == -1)
			{
				textBox1.Text = "异物图片";
				F++;
			}
			textBox6.Text = D.ToString();
			textBox8.Text = F.ToString();
			#endregion

			//DG
			#region
			TestImage(_mimage, svm17);

			if (res == 1)
			{
				textBox1.Text = "毛丝图片";
				D++;
			}
			else if (res == -1)
			{
				textBox1.Text = "油污图片";
				G++;
			}
			textBox6.Text = D.ToString();
			textBox9.Text = G.ToString();
			#endregion

			//EF
			#region
			TestImage(_mimage, svm18);

			if (res == 1)
			{
				textBox1.Text = "小节距图片";
				E++;
			}
			else if (res == -1)
			{
				textBox1.Text = "异物图片";
				F++;
			}
			textBox7.Text = E.ToString();
			textBox8.Text = F.ToString();
			#endregion

			//EG
			#region
			TestImage(_mimage, svm19);

			if (res == 1)
			{
				textBox1.Text = "小节距图片";
				E++;
			}
			else if (res == -1)
			{
				textBox1.Text = "油污图片";
				G++;
			}
			textBox7.Text = E.ToString();
			textBox9.Text = G.ToString();
			#endregion

			//FG
			#region
			TestImage(_mimage, svm20);

			if (res == 1)
			{
				textBox1.Text = "异物图片";
				F++;
			}
			else if (res == -1)
			{
				textBox1.Text = "油污图片";
				G++;
			}
			textBox8.Text = F.ToString();
			textBox9.Text = G.ToString();
			#endregion
		}
		#endregion

		#region 根据已经获得的特征，测试新图片
		private void button29_Click(object sender, EventArgs e)
		{
			//利用XML初始化SVM
			#region
			FileStorage fs0 = new FileStorage("HogFeatures_AB.xml", FileStorage.Mode.Read);
			svm0.Read(fs0.GetFirstTopLevelNode());

			FileStorage fs1 = new FileStorage("HogFeatures_AC.xml", FileStorage.Mode.Read);
			svm1.Read(fs1.GetFirstTopLevelNode());

			FileStorage fs2 = new FileStorage("HogFeatures_AD.xml", FileStorage.Mode.Read);
			svm2.Read(fs2.GetFirstTopLevelNode());

			FileStorage fs3 = new FileStorage("HogFeatures_AE.xml", FileStorage.Mode.Read);
			svm3.Read(fs3.GetFirstTopLevelNode());

			FileStorage fs4 = new FileStorage("HogFeatures_AF.xml", FileStorage.Mode.Read);
			svm4.Read(fs4.GetFirstTopLevelNode());

			FileStorage fs5 = new FileStorage("HogFeatures_AG.xml", FileStorage.Mode.Read);
			svm5.Read(fs5.GetFirstTopLevelNode());

			FileStorage fs6 = new FileStorage("HogFeatures_BC.xml", FileStorage.Mode.Read);
			svm6.Read(fs6.GetFirstTopLevelNode());

			FileStorage fs7 = new FileStorage("HogFeatures_BD.xml", FileStorage.Mode.Read);
			svm7.Read(fs7.GetFirstTopLevelNode());

			FileStorage fs8 = new FileStorage("HogFeatures_BE.xml", FileStorage.Mode.Read);
			svm8.Read(fs8.GetFirstTopLevelNode());

			FileStorage fs9 = new FileStorage("HogFeatures_BF.xml", FileStorage.Mode.Read);
			svm9.Read(fs9.GetFirstTopLevelNode());

			FileStorage fs10 = new FileStorage("HogFeatures_BG.xml", FileStorage.Mode.Read);
			svm10.Read(fs10.GetFirstTopLevelNode());

			FileStorage fs11 = new FileStorage("HogFeatures_CD.xml", FileStorage.Mode.Read);
			svm11.Read(fs11.GetFirstTopLevelNode());

			FileStorage fs12 = new FileStorage("HogFeatures_CE.xml", FileStorage.Mode.Read);
			svm12.Read(fs12.GetFirstTopLevelNode());

			FileStorage fs13 = new FileStorage("HogFeatures_CF.xml", FileStorage.Mode.Read);
			svm13.Read(fs13.GetFirstTopLevelNode());

			FileStorage fs14 = new FileStorage("HogFeatures_CG.xml", FileStorage.Mode.Read);
			svm14.Read(fs14.GetFirstTopLevelNode());

			FileStorage fs15 = new FileStorage("HogFeatures_DE.xml", FileStorage.Mode.Read);
			svm15.Read(fs15.GetFirstTopLevelNode());

			FileStorage fs16 = new FileStorage("HogFeatures_DF.xml", FileStorage.Mode.Read);
			svm16.Read(fs16.GetFirstTopLevelNode());

			FileStorage fs17 = new FileStorage("HogFeatures_DG.xml", FileStorage.Mode.Read);
			svm17.Read(fs17.GetFirstTopLevelNode());

			FileStorage fs18 = new FileStorage("HogFeatures_EF.xml", FileStorage.Mode.Read);
			svm18.Read(fs18.GetFirstTopLevelNode());

			FileStorage fs19 = new FileStorage("HogFeatures_EG.xml", FileStorage.Mode.Read);
			svm19.Read(fs19.GetFirstTopLevelNode());

			FileStorage fs20 = new FileStorage("HogFeatures_FG.xml", FileStorage.Mode.Read);
			svm20.Read(fs20.GetFirstTopLevelNode());

			#endregion//初始化


			//从TXT文件中批量读取测试图像分别进行测试
			StreamReader sin = new StreamReader("info.txt");

			string filename;

			Stopwatch _sw = new Stopwatch();
			_sw.Start();//测试时间

			//清空上次生成的测试结果
			File.WriteAllText("result.txt", string.Empty);

			while ((filename = sin.ReadLine()) != null)
			{
				Image<Bgr, byte> imageput = new Image<Bgr, byte>(new Bitmap(Image.FromFile(filename)));

				MultipleImageTest(imageput);
				++total_count;//每传入一张图片就计数

				//投票，将结果显示在textBox10中	
				#region     
				int[] a = { A, B, C, D, E, F, G };
				int MAX = 0;
				for (int i = 0; i < 7; i++)
				{
					if (a[i] > MAX)
					{
						MAX = a[i];
					}
				}
				if (MAX == A)
				{
					textBox10.Text = "正常图片";
				}
				else if (MAX == B)
				{
					textBox10.Text = "缆扭图片";
				}
				else if (MAX == C)
				{
					textBox10.Text = "露纤图片";
				}
				else if (MAX == D)
				{
					textBox10.Text = "毛丝图片";
				}
				else if (MAX == E)
				{
					textBox10.Text = "小节距图片";
				}
				else if (MAX == F)
				{
					textBox10.Text = "异物图片";
				}
				else if (MAX == G)
				{
					textBox10.Text = "油污图片";
				}
				#endregion

				//将测试结果与原图进行比较
				string Output = null;
				if ((filename[0] == 'L') && (filename[1] == 'N'))//缆扭
				{
					string temp_filename = "缆扭图片";
					if (temp_filename == textBox10.Text.ToString())
					{
						Output = "正确";
						++correct_count;
					}
					else
					{
						Output = "未能识别出正确结果！！！";
					}

				}
				else if ((filename[0] == 'L') && (filename[1] == 'X'))//露纤
				{
					string temp_filename = "露纤图片";
					if (temp_filename == textBox10.Text.ToString())
					{
						Output = "正确";
						++correct_count;
					}
					else
					{
						Output = "未能识别出正确结果！！！";
					}
				}
				else if ((filename[0] == 'M') && (filename[1] == 'S'))//毛丝
				{
					string temp_filename = "毛丝图片";
					if (temp_filename == textBox10.Text.ToString())
					{
						Output = "正确";
						++correct_count;
					}
					else
					{
						Output = "未能识别出正确结果！！！";
					}
				}
				else if ((filename[0] == 'X') && (filename[1] == 'J') && (filename[2] == 'J'))//小节距
				{
					string temp_filename = "小节距图片";
					if (temp_filename == textBox10.Text.ToString())
					{
						Output = "正确";
						++correct_count;
					}
					else
					{
						Output = "未能识别出正确结果！！！";
					}
				}
				else if ((filename[0] == 'Y') && (filename[1] == 'O') && (filename[2] == 'U') && (filename[3] == 'W') && (filename[4] == 'U'))//油污
				{
					string temp_filename = "油污图片";
					if (temp_filename == textBox10.Text.ToString())
					{
						Output = "正确";
						++correct_count;
					}
					else
					{
						Output = "未能识别出正确结果！！！";
					}
				}
				else if ((filename[0] == 'Y') && (filename[1] == 'W'))//毛丝
				{
					string temp_filename = "异物图片";
					if (temp_filename == textBox10.Text.ToString())
					{
						Output = "正确";
						++correct_count;
					}
					else
					{
						Output = "未能识别出正确结果！！！";
					}
				}
				else if ((filename[0] == 'Z') && (filename[1] == 'C'))//正常
				{
					string temp_filename = "正常图片";
					if (temp_filename == textBox10.Text.ToString())
					{
						Output = "正确";
						++correct_count;
					}
					else
					{
						Output = "未能识别出正确结果！！！";
					}
				}
				//将投票得到的MAX值写回到TXT文件
				StreamWriter sw1 = File.AppendText("result.txt");
				string w2 = "原图 : " + filename + "              " + "测试结果 : " + textBox10.Text.ToString() + "           " + Output;
				sw1.WriteLine(w2);
				sw1.Close();
				//清零
				#region               
				A = 0;
				B = 0;
				C = 0;
				D = 0;
				E = 0;
				F = 0;
				G = 0;
				#endregion
			}
			//计算识别的正确率
			correct_recognition_rate = correct_count * 100 / total_count;
			textBox11.Text = correct_recognition_rate.ToString();

			_sw.Stop();
			Console.WriteLine("花费时间：" + _sw.ElapsedMilliseconds / 1000 + "s");//秒

			MessageBox.Show("所有图片已经测试完，测试结果已保存在TXT文件中");

		}
		#endregion
	}
}
