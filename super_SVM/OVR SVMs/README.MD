## 一对一法（one-versus-one,简称OVO SVMs或者pairwise）

　　其做法是在任意两类样本之间设计一个SVM，因此k个类别的样本就需要设计k(k-1)/2个SVM。

　　当对一个未知样本进行分类时，最后得票最多的类别即为该未知样本的类别。

　　Libsvm中的多类分类就是根据这个方法实现的。

　　假设有四类A,B,C,D四类。在训练的时候我选择A,B; A,C; A,D; B,C; B,D;C,D所对应的向量作为训练集，然后得到六个训练结果，在测试的时候，把对应的向量分别对六个结果进行测试，然后采取投票形式，最后得到一组结果。

　　投票是这样的：

　　A=B=C=D=0;

　　(A,B)-classifier 如果是A win,则A=A+1;otherwise,B=B+1;

　　(A,C)-classifier 如果是A win,则A=A+1;otherwise, C=C+1;

　　...

　　(C,D)-classifier 如果是A win,则C=C+1;otherwise,D=D+1;

　　The decision is the Max(A,B,C,D)

评价：这种方法虽然好,但是当类别很多的时候,model的个数是n*(n-1)/2,代价还是相当大的。