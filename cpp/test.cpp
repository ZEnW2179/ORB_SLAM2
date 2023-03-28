#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;

class Tool
{
public:
    /**
     * 构建图像金字塔
     * @param image 输入原图像，这个输入图像所有像素都是有效的，也就是说都是可以在其上提取出FAST角点的
     */
    static vector<Mat> ComputePyramid(cv::Mat image)
    {
        // 调整图像金字塔vector以使得其符合设定的图像层数
        vector<Mat> mvImagePyramid;
        mvImagePyramid.resize(8);

        // 算法生成的图像边
        const int EDGE_THRESHOLD = 19;

        // 开始遍历所有的图层
        for (int level = 0; level < 8; ++level)
        {
            // 获取本层图像的缩放系数
            float scale = 1 / pow(1.2, level);
            // 计算本层图像的像素尺寸大小
            Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
            // 全尺寸图像。包括无效图像区域的大小。将图像进行“补边”，EDGE_THRESHOLD区域外的图像不进行FAST角点检测
            Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
            // 定义了两个变量：temp是扩展了边界的图像，masktemp并未使用
            Mat temp(wholeSize, image.type()), masktemp;
            // mvImagePyramid 刚开始时是个空的vector<Mat>
            // 把图像金字塔该图层的图像指针mvImagePyramid指向temp的中间部分（这里为浅拷贝，内存相同）
            mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

            // Compute the resized image
            // 计算第0层以上resize后的图像
            if (level != 0)
            {
                // 将上一层金字塔图像根据设定sz缩放到当前层级
                resize(mvImagePyramid[level - 1], // 输入图像
                       mvImagePyramid[level],     // 输出图像
                       sz,                        // 输出图像的尺寸
                       0,                         // 水平方向上的缩放系数，留0表示自动计算
                       0,                         // 垂直方向上的缩放系数，留0表示自动计算
                       cv::INTER_LINEAR);         // 图像缩放的差值算法类型，这里的是线性插值算法

                // //!  原代码mvImagePyramid 并未扩充，上面resize应该改为如下
                // resize(image,	                //输入图像
                // 	   mvImagePyramid[level], 	//输出图像
                // 	   sz, 						//输出图像的尺寸
                // 	   0, 						//水平方向上的缩放系数，留0表示自动计算
                // 	   0,  						//垂直方向上的缩放系数，留0表示自动计算
                // 	   cv::INTER_LINEAR);		//图像缩放的差值算法类型，这里的是线性插值算法

                // 把源图像拷贝到目的图像的中央，四面填充指定的像素。图片如果已经拷贝到中间，只填充边界
                // 这样做是为了能够正确提取边界的FAST角点
                // EDGE_THRESHOLD指的这个边界的宽度，由于这个边界之外的像素不是原图像素而是算法生成出来的，所以不能够在EDGE_THRESHOLD之外提取特征点
                copyMakeBorder(mvImagePyramid[level],                 // 源图像
                               temp,                                  // 目标图像（此时其实就已经有大了一圈的尺寸了）
                               EDGE_THRESHOLD, EDGE_THRESHOLD,        // top & bottom 需要扩展的border大小
                               EDGE_THRESHOLD, EDGE_THRESHOLD,        // left & right 需要扩展的border大小
                               BORDER_REFLECT_101 + BORDER_ISOLATED); // 扩充方式，opencv给出的解释：

                /*Various border types, image boundaries are denoted with '|'
                 * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
                 * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
                 * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
                 * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
                 * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
                 */

                // BORDER_ISOLATED	表示对整个图像进行操作
                //  https://docs.opencv.org/3.4.4/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36
            }
            else
            {
                // 对于第0层未缩放图像，直接将图像深拷贝到temp的中间，并且对其周围进行边界扩展。此时temp就是对原图扩展后的图像
                copyMakeBorder(image, // 这里是原图像
                               temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101);
            }
            // //! 原代码mvImagePyramid 并未扩充，应该添加下面一行代码
            // mvImagePyramid[level] = temp;
        }
        return mvImagePyramid;
    }
};

int main()
{
    Mat img1 = imread("/home/SLAM/Code/cpp/left.png");
    Mat img2 = imread("/home/SLAM/Code/cpp/right.png");
    vector<KeyPoint> keypoint1, keypoint2;
    Mat descriptor1, descriptor2;
    Mat result_img;

    Ptr<ORB> detector = ORB::create();
    // 提取特征点
    detector->detect(img1, keypoint1, noArray());
    detector->detect(img2, keypoint2, noArray());
    // 计算特征点的描述子
    detector->compute(img1, keypoint1, descriptor1);
    detector->compute(img2, keypoint2, descriptor2);
    // 计算左右图的影像金字塔（共8层）
    vector<Mat> pyramid1 = Tool::ComputePyramid(img1);
    vector<Mat> pyramid2 = Tool::ComputePyramid(img2);

    // 要用到的一些阈值
    const int TH_HIGH = 100;
    const int TH_LOW = 50;
    const int HISTO_LENGTH = 30;

    // 为匹配结果预先分配内存，数据类型为float型
    // mvuRight存储右图匹配点索引
    // mvDepth存储特征点的深度信息
    // mvIdxR存储右图匹配点idx
    vector<float> mvuRight = vector<float>(keypoint1.size(), -1.0f);
    vector<float> mvDepth = vector<float>(keypoint1.size(), -1.0f);
    vector<pair<int, int>> mvIdxR;

    // orb特征相似度阈值  -> mean ～= (max  + min) / 2
    const int thOrbDist = (TH_HIGH + TH_LOW) / 2;

    // ORB图像金字塔0层（原始图形）行数
    const int nRows = img1.rows;

    // 建立一个二维向量存储每一行的orb特征点的列坐标的索引
    //  vRowIndices[0] = [1，2，5，8, 11]   第1行有5个特征点,他们的列号（即x坐标）分别是1,2,5,8,11
    //  vRowIndices[1] = [2，6，7，9, 13, 17, 20]  第2行有7个特征点
    vector<vector<size_t>> vRowIndices(nRows, vector<size_t>());
    for (int i = 0; i < nRows; i++)
    {
        vRowIndices[i].reserve(200); // 每一行预留200个位置
    }

    // Step 1. 行特征点统计。 考虑用图像金字塔尺度作为偏移，左图中对应右图的一个特征点可能存在于多行，而非唯一的一行
    for (int iR = 0; iR < keypoint2.size(); iR++)
    {
        // 获取右图特征点ir的y坐标，即行号
        const KeyPoint &kp = keypoint2[iR];
        const float &kpY = kp.pt.y;

        // 计算特征点ir在行方向上，可能的偏移范围r，即可能的行号为[kpY + r, kpY -r]
        // 表示在全尺寸(scale = 1)的情况下，假设有2个像素的偏移，随着尺度变化，r也跟着变化
        const float r = 2.0f * pow(1.2, keypoint2[iR].octave);
        const int maxr = ceil(kpY + r);
        const int minr = floor(kpY - r);

        // 将特征点ir保证在可能的行号中
        for (int yi = minr; yi < maxr; yi++)
        {
            vRowIndices[yi].push_back(iR);
        }
    }

    // 下面是 粗匹配 + 精匹配的过程
    // 对于立体矫正后的两张图，在列方向(x)存在最大视差maxd和最小视差mind
    // 也即是左图中任何一点p，在右图上的匹配点的范围为应该是[p - maxd, p - mind], 而不需要遍历每一行所有的像素
    // maxd = baseline * length_focal / minZ
    // mind = baseline * length_focal / maxZ

    const float mbf = 47.90639384423901;
    const float minD = 0;                 // 最小视差为0，对应无穷远
    const float maxD = 435.2046959714599; // 最大视差对应的距离是相机的焦距，单位mm

    // 保存sad块匹配相似度和左图特征点索引
    vector<pair<int, int>> vDistIdx;
    vDistIdx.reserve(keypoint1.size());

    // 为左图每一个特征点il，在右图搜索最相似的特征点ir
    for (int iL = 0; iL < keypoint1.size(); iL++)
    {
        const KeyPoint &kpL = keypoint1[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        // 获取左图特征点il所在行，以及在右图对应行中可能的匹配点
        const vector<size_t> &vCandidates = vRowIndices[vL];
        if (vCandidates.empty())
            continue;

        // 计算理论上的最佳搜索范围
        const float minU = uL - maxD;
        const float maxU = uL - minD;

        // 最大搜索范围小于0，说明无匹配点
        if (maxU < 0)
            continue;

        // 初始化最佳相似度，用最大相似度，以及最佳匹配点索引
        int bestDist = TH_HIGH;
        size_t bestIdxR = 0;
        const Mat &dL = descriptor1.row(iL);

        // Step 2. 粗配准。左图特征点il与右图中的可能的匹配点进行逐个比较,得到最相似匹配点的描述子距离和索引
        for (size_t iC = 0; iC < vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const KeyPoint &kpR = keypoint2[iR];

            // 左图特征点il与待匹配点ic的空间尺度差超过2，放弃
            if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                continue;

            // 使用列坐标(x)进行匹配，和stereomatch一样
            const float &uR = kpR.pt.x;

            // 超出理论搜索范围[minU, maxU]，可能是误匹配，放弃
            if (uR > minU && uR <= maxU)
            {
                const Mat &dR = descriptor2.row(iR);

                // 计算左右特征点描述子汉明距离
                const int *pa = dL.ptr<int32_t>();
                const int *pb = dR.ptr<int32_t>();

                int dist = 0;
                // 8*32=256bit
                for (int i = 0; i < 8; i++, pa++, pb++)
                {
                    unsigned int v = *pa ^ *pb; // 相等为0,不等为1
                    // 下面的操作就是计算其中bit为1的个数了,这个操作看上面的链接就好
                    // 其实我觉得也还阔以直接使用8bit的查找表,然后做32次寻址操作就完成了;不过缺点是没有利用好CPU的字长
                    v = v - ((v >> 1) & 0x55555555);
                    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
                    dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
                }

                // 统计最小相似度及其对应的列坐标(x)
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Step 3. 图像块滑动窗口用SAD(Sum of absolute differences，差的绝对和)实现精确匹配.
        if (bestDist < thOrbDist)
        {
            // 如果刚才匹配过程中的最佳描述子距离小于给定的阈值
            // 计算右图特征点x坐标和对应的金字塔尺度
            const float uR0 = keypoint2[bestIdxR].pt.x;
            const float scaleFactor = 1 / pow(1.2, kpL.octave);

            // 尺度缩放后的左右图特征点坐标
            const float scaleduL = round(kpL.pt.x * scaleFactor);
            const float scaledvL = round(kpL.pt.y * scaleFactor);
            const float scaleduR0 = round(uR0 * scaleFactor);

            // 滑动窗口搜索, 类似模版卷积或滤波
            // w表示sad相似度的窗口半径
            const int w = 5;

            // 提取左图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像块patch
            Mat IL = pyramid1[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduL - w, scaleduL + w + 1);
            IL.convertTo(IL, CV_32F);

            // 图像块均值归一化，降低亮度变化对相似度计算的影响
            // IL = IL - IL.at<float>(w, w) * Mat::ones(IL.rows, IL.cols, CV_32F);

            // 初始化最佳相似度
            int bestDist = INT_MAX;

            // 通过滑动窗口搜索优化，得到的列坐标偏移量
            int bestincR = 0;

            // 滑动窗口的滑动范围为（-L, L）
            const int L = 5;

            // 初始化存储图像块相似度
            vector<float> vDists;
            vDists.resize(2 * L + 1);

            // 计算滑动窗口滑动范围的边界，因为是块匹配，还要算上图像块的尺寸
            // 列方向起点 iniu = r0 - 最大窗口滑动范围 - 图像块尺寸
            // 列方向终点 eniu = r0 + 最大窗口滑动范围 + 图像块尺寸 + 1
            // 此次 + 1 和下面的提取图像块是列坐标+1是一样的，保证提取的图像块的宽是2 * w + 1
            // ! 源码： const float iniu = scaleduR0+L-w; 错误
            // scaleduR0：右图特征点x坐标
            const float iniu = scaleduR0 - L - w;
            const float endu = scaleduR0 + L + w + 1;

            // 判断搜索是否越界
            if (iniu < 0 || endu >= pyramid2[kpL.octave].cols)
                continue;

            // 在搜索范围内从左到右滑动，并计算图像块相似度
            for (int incR = -L; incR <= +L; incR++)
            {
                // 提取右图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
                cv::Mat IR = pyramid2[kpL.octave].rowRange(scaledvL - w, scaledvL + w + 1).colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
                IR.convertTo(IR, CV_32F);

                // 图像块均值归一化，降低亮度变化对相似度计算的影响
                // IR = IR - IR.at<float>(w, w) * Mat::ones(IR.rows, IR.cols, CV_32F);

                // sad 计算，值越小越相似
                float dist = cv::norm(IL, IR, cv::NORM_L2);

                // 统计最小sad和偏移量
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestincR = incR;
                }

                // L+incR 为refine后的匹配点列坐标(x)
                vDists[L + incR] = dist;
            }

            // 搜索窗口越界判断
            if (bestincR == -L || bestincR == L)
                continue;

            // Step 4. 亚像素插值, 使用最佳匹配点及其左右相邻点构成抛物线来得到最小sad的亚像素坐标
            // 使用3点拟合抛物线的方式，用极小值代替之前计算的最优是差值
            //    \                 / <- 由视差为14，15，16的相似度拟合的抛物线
            //      .             .(16)
            //         .14     .(15) <- int/uchar最佳视差值
            //              .
            //           （14.5）<- 真实的视差值
            //   deltaR = 15.5 - 16 = -0.5
            // 公式参考opencv sgbm源码中的亚像素插值公式
            // 或论文<<On Building an Accurate Stereo Matching System on Graphics Hardware>> 公式7

            const float dist1 = vDists[L + bestincR - 1];
            const float dist2 = vDists[L + bestincR];
            const float dist3 = vDists[L + bestincR + 1];
            const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

            // 亚像素精度的修正量应该是在[-1,1]之间，否则就是误匹配
            if (deltaR < -1 || deltaR > 1)
                continue;

            // 根据亚像素精度偏移量delta调整最佳匹配索引
            float bestuR = pow(1.2, kpL.octave) * ((float)scaleduR0 + (float)bestincR + deltaR);
            float disparity = (uL - bestuR);
            if (disparity >= minD && disparity < maxD)
            {
                // 如果存在负视差，则约束为0.01
                if (disparity <= 0)
                {
                    disparity = 0.01;
                    bestuR = uL - 0.01;
                }

                // 根据视差值计算深度信息
                // 保存最相似点的列坐标(x)信息
                // 保存归一化sad最小相似度
                // Step 5. 最优视差值/深度选择.
                mvDepth[iL] = mbf / disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int, int>(bestDist, iL));
                mvIdxR.push_back(pair<int, int>(iL, bestIdxR));
            }
        }
        // Step 6. 删除离群点(outliers)
        // 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是匹配的，比如光照变化、弱纹理、无纹理等同样会造成误匹配
        // 误匹配判断条件  norm_sad > 1.5 * 1.4 * median
        sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size() / 2].first;
        const float thDist = 1.5f * 1.4f * median;

        for (int i = vDistIdx.size() - 1; i >= 0; i--)
        {
            if (vDistIdx[i].first < thDist)
            {
                break;
            }
            else
            {
                // 误匹配点置为-1，和初始化时保持一直，作为error code
                mvuRight[vDistIdx[i].second] = -1;
                mvDepth[vDistIdx[i].second] = -1;
            }
        }
    }

    for (size_t i = 0; i < mvIdxR.size(); i++)
    {
        cout << mvIdxR[i].first << " " << mvIdxR[i].second << "\n";
    }

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    vector<DMatch> matches;
    matcher->match(descriptor1, descriptor2, matches);

    int num = mvIdxR.size();
    nth_element(matches.begin(), matches.begin() + num, matches.end());
    matches.erase(matches.begin() + num, matches.end());

    for (size_t i = 0; i < matches.size(); i++)
    {
        matches[i].queryIdx = mvIdxR[i].first;
        matches[i].trainIdx = mvIdxR[i].second;
        matches[i].distance = 0;
        matches[i].imgIdx = 0;
    }

    drawMatches(img1, keypoint1, img2, keypoint2, matches, result_img);
    drawKeypoints(img1, keypoint1, img1);
    drawKeypoints(img2, keypoint2, img2);
    imshow("result_img", result_img);
    waitKey(0);
    system("pause");
    return 0;
}