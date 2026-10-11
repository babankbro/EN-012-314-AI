export interface ModuleData {
  id: string;
  week: string;
  titleTh: string;
  titleEn: string;
  description: string;
  clo: string;
  llos: string[];
  topics: string[];
  labTitle: string;
  reviewSheet: string;
  icon: string;
  color: string;
}

// 10 บทเรียน จับคู่กับแผนการสอนรายสัปดาห์ใน มคอ.3 (สัปดาห์ 1–13)
export const modules: ModuleData[] = [
  {
    id: "00",
    week: "1",
    titleTh: "ลำดับชั้นของปัญญาประดิษฐ์",
    titleEn: "The AI Hierarchy",
    description:
      "AI, Machine Learning, Neural Network, Deep Learning, Generative AI และ LLM พร้อมสาขาย่อยของ AI เช่น Planning, Automated Reasoning, Knowledge Representation, Computer Vision และ AI Agent",
    clo: "CLO1",
    llos: ["LLO1-1", "LLO1-3"],
    topics: ["AI / ML / NN / DL / GenAI / LLM", "สาขาย่อยของ AI", "Planning & Scheduling", "Automated Reasoning", "Computer Vision", "AI Agent", "GAN", "Transformer"],
    labTitle: "จำแนกระดับและสาขาของงาน AI",
    reviewSheet: "review_sheet_00.html",
    icon: "Network",
    color: "slate",
  },
  {
    id: "01",
    week: "2",
    titleTh: "ภาพรวม Machine Learning",
    titleEn: "The Machine Learning Landscape",
    description:
      "ประเภทของการเรียนรู้ (Supervised, Unsupervised, Semi-supervised, Reinforcement) ความท้าทายหลัก และการแบ่งข้อมูล Train/Validation/Test",
    clo: "CLO1",
    llos: ["LLO1-1", "LLO1-2"],
    topics: ["ประเภทของ ML", "Batch vs Online", "Instance vs Model-based", "Overfitting / Underfitting", "Validation & Cross-Validation"],
    labTitle: "จับคู่ปัญหากับรูปแบบการเรียนรู้",
    reviewSheet: "review_sheet_01.html",
    icon: "Compass",
    color: "blue",
  },
  {
    id: "02",
    week: "3",
    titleTh: "ML Pipeline และการเตรียมข้อมูล",
    titleEn: "ML Pipeline & Data Preparation",
    description:
      "ขั้นตอนการพัฒนาโมเดลตั้งแต่โหลดข้อมูล สำรวจ (EDA) ทำความสะอาด ปรับสเกล และแบ่ง Train/Test ด้วย pandas และ scikit-learn",
    clo: "CLO2",
    llos: ["LLO2-1", "LLO2-2"],
    topics: ["Data acquisition", "EDA & Cleaning", "Feature selection", "Scaling", "Train/Test split", "Pipeline"],
    labTitle: "เตรียมข้อมูลและ Train/Test Split (Diabetes)",
    reviewSheet: "review_sheet_02.html",
    icon: "Workflow",
    color: "green",
  },
  {
    id: "03",
    week: "4",
    titleTh: "การจำแนกประเภทและการประเมินผล",
    titleEn: "Classification & Model Evaluation",
    description:
      "Logistic Regression, KNN, SVM, Decision Tree, Random Forest พร้อม Confusion Matrix, Accuracy, Precision, Recall และ F1-score",
    clo: "CLO2",
    llos: ["LLO2-2", "LLO2-3"],
    topics: ["Logistic Regression", "KNN", "SVM", "Decision Tree", "Ensemble / Random Forest", "Confusion Matrix", "Precision / Recall / F1"],
    labTitle: "Classification และการคำนวณ Metrics",
    reviewSheet: "review_sheet_03.html",
    icon: "Tags",
    color: "yellow",
  },
  {
    id: "04",
    week: "5",
    titleTh: "การถดถอยและการประเมินโมเดล",
    titleEn: "Regression & Model Evaluation",
    description:
      "Linear Regression, Gradient Descent, Regularization, SVR และ Tree-based Regressor กับข้อมูล California Housing ประเมินด้วย MSE, RMSE, R²",
    clo: "CLO2",
    llos: ["LLO2-2", "LLO2-3"],
    topics: ["Linear Regression", "Gradient Descent", "Ridge / Lasso", "SVR", "Normalization / Standardization", "MSE / RMSE / R²"],
    labTitle: "Regression ด้วย scikit-learn (California Housing)",
    reviewSheet: "review_sheet_04.html",
    icon: "TrendingUp",
    color: "red",
  },
  {
    id: "05",
    week: "6",
    titleTh: "Clustering, การลดมิติ และ Anomaly Detection",
    titleEn: "Clustering, Dimensionality Reduction & Anomaly Detection",
    description:
      "K-Means, DBSCAN, PCA, t-SNE และ Gaussian Mixture สำหรับตรวจจับความผิดปกติ พร้อมกรณีศึกษา MNIST แบบไม่มีผู้สอน",
    clo: "CLO2",
    llos: ["LLO2-2", "LLO2-3"],
    topics: ["K-Means", "Elbow / Silhouette", "DBSCAN", "PCA", "t-SNE", "GMM Anomaly Detection"],
    labTitle: "K-Means, PCA และการเลือกอัลกอริทึม",
    reviewSheet: "review_sheet_05.html",
    icon: "Boxes",
    color: "purple",
  },
  {
    id: "06",
    week: "9",
    titleTh: "Neural Networks และ Deep Learning",
    titleEn: "Neural Networks & Deep Learning",
    description:
      "Perceptron, MLP, Backpropagation, Activation, Loss, Optimizer และเทคนิคการฝึก Deep Neural Network ด้วย Keras",
    clo: "CLO3",
    llos: ["LLO3-1"],
    topics: ["Perceptron / MLP", "Backpropagation", "Activation functions", "Loss & Optimizer", "Vanishing gradients", "Dropout / BatchNorm"],
    labTitle: "ออกแบบและฝึก MLP ด้วย Keras",
    reviewSheet: "review_sheet_06.html",
    icon: "BrainCircuit",
    color: "teal",
  },
  {
    id: "07",
    week: "10",
    titleTh: "CNN และการจำแนกภาพ",
    titleEn: "Convolutional Neural Networks & Image Classification",
    description:
      "Convolution, Pooling, สถาปัตยกรรม CNN ชื่อดัง และขั้นตอนอ่านภาพ แปลงข้อมูล สร้างโมเดล ฝึก ทำนาย และประเมินผลด้วย MNIST",
    clo: "CLO3",
    llos: ["LLO3-1", "LLO3-2"],
    topics: ["Convolution & Filters", "Pooling", "LeNet / AlexNet / ResNet", "Image preprocessing", "One-hot encoding", "Train / Predict / Evaluate"],
    labTitle: "Image Classification ด้วย CNN",
    reviewSheet: "review_sheet_07.html",
    icon: "ScanEye",
    color: "indigo",
  },
  {
    id: "08",
    week: "12",
    titleTh: "การประมวลผลและทำความเข้าใจภาษาธรรมชาติ",
    titleEn: "Natural Language Processing & Understanding",
    description:
      "NLP/NLU pipeline ตั้งแต่ทำความสะอาดข้อความ ตัดคำ สร้างฟีเจอร์ (BoW, TF-IDF) จนถึง Sentiment Analysis และบทบาทของ LLM",
    clo: "CLO4",
    llos: ["LLO4-1", "LLO4-2"],
    topics: ["NLP / NLU pipeline", "Tokenization (ไทย/อังกฤษ)", "Stop words", "BoW / TF-IDF", "Text classification", "Traditional NLP vs LLM"],
    labTitle: "Text Classification และ Sentiment Analysis",
    reviewSheet: "review_sheet_08.html",
    icon: "MessageSquareText",
    color: "orange",
  },
  {
    id: "09",
    week: "13",
    titleTh: "Reinforcement Learning และ Game Playing",
    titleEn: "Reinforcement Learning & Game Playing",
    description:
      "Agent, Environment, State, Action, Reward, MDP, Q-Learning, Exploration/Exploitation และ Deep Q-Network กับ CartPole, Maze และ Atari",
    clo: "CLO4",
    llos: ["LLO4-3"],
    topics: ["Agent / Environment / Reward", "MDP", "Q-Learning & Bellman", "ε-greedy", "DQN", "CartPole / Maze / Atari"],
    labTitle: "Q-Learning กับเกม Grid World",
    reviewSheet: "review_sheet_09.html",
    icon: "Gamepad2",
    color: "rose",
  },
];

export function getModuleById(id: string): ModuleData | undefined {
  return modules.find((m) => m.id === id);
}

export const colorMap: Record<string, { bg: string; text: string; border: string; accent: string }> = {
  slate: { bg: "bg-slate-100 dark:bg-slate-800/40", text: "text-slate-700 dark:text-slate-300", border: "border-slate-200 dark:border-slate-700", accent: "from-slate-500 to-gray-500" },
  blue: { bg: "bg-blue-50 dark:bg-blue-900/20", text: "text-blue-700 dark:text-blue-300", border: "border-blue-200 dark:border-blue-800", accent: "from-blue-500 to-cyan-500" },
  green: { bg: "bg-green-50 dark:bg-green-900/20", text: "text-green-700 dark:text-green-300", border: "border-green-200 dark:border-green-800", accent: "from-green-500 to-emerald-500" },
  yellow: { bg: "bg-yellow-50 dark:bg-yellow-900/20", text: "text-yellow-700 dark:text-yellow-300", border: "border-yellow-200 dark:border-yellow-800", accent: "from-yellow-500 to-orange-500" },
  red: { bg: "bg-red-50 dark:bg-red-900/20", text: "text-red-700 dark:text-red-300", border: "border-red-200 dark:border-red-800", accent: "from-red-500 to-pink-500" },
  purple: { bg: "bg-purple-50 dark:bg-purple-900/20", text: "text-purple-700 dark:text-purple-300", border: "border-purple-200 dark:border-purple-800", accent: "from-purple-500 to-violet-500" },
  teal: { bg: "bg-teal-50 dark:bg-teal-900/20", text: "text-teal-700 dark:text-teal-300", border: "border-teal-200 dark:border-teal-800", accent: "from-teal-500 to-cyan-500" },
  indigo: { bg: "bg-indigo-50 dark:bg-indigo-900/20", text: "text-indigo-700 dark:text-indigo-300", border: "border-indigo-200 dark:border-indigo-800", accent: "from-indigo-500 to-blue-500" },
  orange: { bg: "bg-orange-50 dark:bg-orange-900/20", text: "text-orange-700 dark:text-orange-300", border: "border-orange-200 dark:border-orange-800", accent: "from-orange-500 to-amber-500" },
  rose: { bg: "bg-rose-50 dark:bg-rose-900/20", text: "text-rose-700 dark:text-rose-300", border: "border-rose-200 dark:border-rose-800", accent: "from-rose-500 to-pink-500" },
};
