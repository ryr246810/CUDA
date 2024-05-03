
#include <bessel.hxx>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace std;

double Bessjy::j0(const double x) 
{
  if ((ax=abs(x)) < 8.0) {
    rat(x,j0r,j0s,6);
    return nump*(y-xj00)*(y-xj10)/denp;
  } else {
    asp(j0pn,j0pd,j0qn,j0qd,1.);
    return sqrt(twoopi/ax)*(cos(xx)*nump/denp-z*sin(xx)*numq/denq);
  }
}


double Bessjy::j1(const double x) 
{
  if ((ax=abs(x)) < 8.0) {
    rat(x,j1r,j1s,6);
    return x*nump*(y-xj01)*(y-xj11)/denp;
  } else {
    asp(j1pn,j1pd,j1qn,j1qd,3.);
    double ans=sqrt(twoopi/ax)*(cos(xx)*nump/denp-z*sin(xx)*numq/denq);
    return x > 0.0 ? ans : -ans;
  }
}


double Bessjy::y0(const double x) 
{
  if (x < 8.0) {
    double j0x = j0(x);
    rat(x,y0r,y0s,8);
    return nump/denp+twoopi*j0x*log(x);
  } else {
    ax=x;
    asp(y0pn,y0pd,y0qn,y0qd,1.);
    return sqrt(twoopi/x)*(sin(xx)*nump/denp+z*cos(xx)*numq/denq);
  }
}


double Bessjy::y1(const double x) 
{
  if (x < 8.0) {
    double j1x = j1(x);
    rat(x,y1r,y1s,7);
    return x*nump/denp+twoopi*(j1x*log(x)-1.0/x);
  } else {
    ax=x;
    asp(y1pn,y1pd,y1qn,y1qd,3.);
    return sqrt(twoopi/x)*(sin(xx)*nump/denp+z*cos(xx)*numq/denq);
  }
}


void Bessjy::rat(const double x, const double *r, const double *s, const int n) 
{
  y = x*x;
  z=64.0-y;
  nump=r[n];
  denp=s[n];
  for (int i=n-1;i>=0;i--) {
    nump=nump*z+r[i];
    denp=denp*y+s[i];
  }
}


void Bessjy::asp(const double *pn, const double *pd, const double *qn, const double *qd, const double fac) 
{
  z=8.0/ax;
  y=z*z;
  xx=ax-fac*pio4;
  nump=pn[4];
  denp=pd[4];
  numq=qn[4];
  denq=qd[4];
  for (int i=3;i>=0;i--) {
    nump=nump*y+pn[i];
    denp=denp*y+pd[i];
    numq=numq*y+qn[i];
    denq=denq*y+qd[i];
  }	
}


double Bessjy::jn(const int n, const double x)
{
  const double ACC=160.0;
  const int IEXP=numeric_limits<double>::max_exponent/2;
  bool jsum;
  int j,k,m;
  double ax,bj,bjm,bjp,dum,sum,tox,ans;
  if (n==0) return j0(x);
  if (n==1) return j1(x);
  ax=abs(x);
  if (ax*ax <= 8.0*numeric_limits<double>::min()) return 0.0;
  else if (ax > double(n)) {
    tox=2.0/ax;
    bjm=j0(ax);
    bj=j1(ax);
    for (j=1;j<n;j++) {
      bjp=j*tox*bj-bjm;
      bjm=bj;
      bj=bjp;
    }
    ans=bj;
  } else {
    tox=2.0/ax;
    m=2*((n+int(sqrt(ACC*n)))/2);
    jsum=false;
    bjp=ans=sum=0.0;
    bj=1.0;
    for (j=m;j>0;j--) {
      bjm=j*tox*bj-bjp;
      bjp=bj;
      bj=bjm;
      dum=frexp(bj,&k);
      if (k > IEXP) {
	bj=ldexp(bj,-IEXP);
	bjp=ldexp(bjp,-IEXP);
	ans=ldexp(ans,-IEXP);
	sum=ldexp(sum,-IEXP);
      }
      if (jsum) sum += bj;
      jsum=!jsum;
      if (j == n) ans=bjp;
    }
    sum=2.0*sum-bj;
    ans /= sum;
  }
  return x < 0.0 && (n & 1) ? -ans : ans;
}


double Bessjy::yn(const int n, const double x)
{
  int j;
  double by,bym,byp,tox;
  if (n==0) return y0(x);
  if (n==1) return y1(x);
  tox=2.0/x;
  by=y1(x);
  bym=y0(x);
  for (j=1;j<n;j++) {
    byp=j*tox*by-bym;
    bym=by;
    by=byp;
  }
  return by;
}


double Bessik::i0(const double x) 
{
  if ((ax=abs(x)) < 15.0) {
    y = x*x;
    return poly(i0p,13,y)/poly(i0q,4,225.-y);
  } else {
    z=1.0-15.0/ax;
    return exp(ax)*poly(i0pp,4,z)/(poly(i0qq,5,z)*sqrt(ax));
  }
}


double Bessik::i1(const double x) 
{
  if ((ax=abs(x)) < 15.0) {
    y=x*x;
    return x*poly(i1p,13,y)/poly(i1q,4,225.-y);
  } else {
    z=1.0-15.0/ax;
    double ans=exp(ax)*poly(i1pp,4,z)/(poly(i1qq,5,z)*sqrt(ax));
    return x > 0.0 ? ans : -ans;
  }
}


double Bessik::k0(const double x) 
{
  if (x <= 1.0) {
    z=x*x;
    term = poly(k0pi,4,z)*log(x)/poly(k0qi,2,1.-z);
    return poly(k0p,4,z)/poly(k0q,2,1.-z)-term;
  } else {
    z=1.0/x;
    return exp(-x)*poly(k0pp,7,z)/(poly(k0qq,7,z)*sqrt(x));
  }
}


double Bessik::k1(const double x) 
{
  if (x <= 1.0) {
    z=x*x;
    term = poly(k1pi,4,z)*log(x)/poly(k1qi,2,1.-z);
    return x*(poly(k1p,4,z)/poly(k1q,2,1.-z)+term)+1./x;
  } else {
    z=1.0/x;
    return exp(-x)*poly(k1pp,7,z)/(poly(k1qq,7,z)*sqrt(x));
  }
}


double Bessik::kn(const int n, const double x)
{
  int j;
  double bk,bkm,bkp,tox;
  if (n==0) return k0(x);
  if (n==1) return k1(x);
  tox=2.0/x;
  bkm=k0(x);
  bk=k1(x);
  for (j=1;j<n;j++) {
    bkp=bkm+j*tox*bk;
    bkm=bk;
    bk=bkp;
  }
  return bk;
}


double Bessik::in(const int n, const double x)
{
  const double ACC=200.0;
  const int IEXP=numeric_limits<double>::max_exponent/2;
  int j,k;
  double bi,bim,bip,dum,tox,ans;
  if (n==0) return i0(x);
  if (n==1) return i1(x);
  if (x*x <= 8.0*numeric_limits<double>::min()) return 0.0;
  else {
    tox=2.0/abs(x);
    bip=ans=0.0;
    bi=1.0;
    for (j=2*(n+int(sqrt(ACC*n)));j>0;j--) {
      bim=bip+j*tox*bi;
      bip=bi;
      bi=bim;
      dum=frexp(bi,&k);
      if (k > IEXP) {
	ans=ldexp(ans,-IEXP);
	bi=ldexp(bi,-IEXP);
	bip=ldexp(bip,-IEXP);
      }
      if (j == n) ans=bip;
    }
    ans *= i0(x)/bi;
    return x < 0.0 && (n & 1) ? -ans : ans;
  }
}


const double Bessjy::xj00=5.783185962946785;
const double Bessjy::xj10=3.047126234366209e1;
const double Bessjy::xj01=1.468197064212389e1;
const double Bessjy::xj11=4.921845632169460e1;
const double Bessjy::twoopi=0.6366197723675813;
const double Bessjy::pio4=0.7853981633974483;
const double Bessjy::j0r[]={1.682397144220462e-4,2.058861258868952e-5,
			    5.288947320067750e-7,5.557173907680151e-9,2.865540042042604e-11,
			    7.398972674152181e-14,7.925088479679688e-17};
const double Bessjy::j0s[]={1.0,1.019685405805929e-2,5.130296867064666e-5,
			    1.659702063950243e-7,3.728997574317067e-10,
			    5.709292619977798e-13,4.932979170744996e-16};
const double Bessjy::j0pn[]={9.999999999999999e-1,1.039698629715637,
			     2.576910172633398e-1,1.504152485749669e-2,1.052598413585270e-4};
const double Bessjy::j0pd[]={1.0,1.040797262528109,2.588070904043728e-1,
			     1.529954477721284e-2,1.168931211650012e-4};
const double Bessjy::j0qn[]={-1.562499999999992e-2,-1.920039317065641e-2,
			     -5.827951791963418e-3,-4.372674978482726e-4,-3.895839560412374e-6};
const double Bessjy::j0qd[]={1.0,1.237980436358390,3.838793938147116e-1,
			     3.100323481550864e-2,4.165515825072393e-4};
const double Bessjy::j1r[]={7.309637831891357e-5,3.551248884746503e-6,
			    5.820673901730427e-8,4.500650342170622e-10,1.831596352149641e-12,
			    3.891583573305035e-15,3.524978592527982e-18};
const double Bessjy::j1s[]={1.0,9.398354768446072e-3,4.328946737100230e-5,
			    1.271526296341915e-7,2.566305357932989e-10,
			    3.477378203574266e-13,2.593535427519985e-16};
const double Bessjy::j1pn[]={1.0,1.014039111045313,2.426762348629863e-1,
			     1.350308200342000e-2,9.516522033988099e-5};
const double Bessjy::j1pd[]={1.0,1.012208056357845,2.408580305488938e-1,
			     1.309511056184273e-2,7.746422941504713e-5};
const double Bessjy::j1qn[]={4.687499999999991e-2,5.652407388406023e-2,
			     1.676531273460512e-2,1.231216817715814e-3,1.178364381441801e-5};
const double Bessjy::j1qd[]={1.0,1.210119370463693,3.626494789275638e-1,
			     2.761695824829316e-2,3.240517192670181e-4};
const double Bessjy::y0r[]={-7.653778457189104e-3,-5.854760129990403e-2,
			    3.720671300654721e-4,3.313722284628089e-5,4.247761237036536e-8,
			    -4.134562661019613e-9,-3.382190331837473e-11,
			    -1.017764126587862e-13,-1.107646382675456e-16};
const double Bessjy::y0s[]={1.0,1.125494540257841e-2,6.427210537081400e-5,
			    2.462520624294959e-7,7.029372432344291e-10,1.560784108184928e-12,
			    2.702374957564761e-15,3.468496737915257e-18,2.716600180811817e-21};
const double Bessjy::y0pn[]={9.999999999999999e-1,1.039698629715637,
			     2.576910172633398e-1,1.504152485749669e-2,1.052598413585270e-4};
const double Bessjy::y0pd[]={1.0,1.040797262528109,2.588070904043728e-1,
			     1.529954477721284e-2,1.168931211650012e-4};
const double Bessjy::y0qn[]={-1.562499999999992e-2,-1.920039317065641e-2,
			     -5.827951791963418e-3,-4.372674978482726e-4,-3.895839560412374e-6};
const double Bessjy::y0qd[]={1.0,1.237980436358390,3.838793938147116e-1,
			     3.100323481550864e-2,4.165515825072393e-4};
const double Bessjy::y1r[]={-1.041835425863234e-1,-1.135093963908952e-5,
			    2.212118520638132e-4,1.270981874287763e-6,
			    -3.982892100836748e-8,-4.820712110115943e-10,
			    -1.929392690596969e-12,-2.725259514545605e-15};
const double Bessjy::y1s[]={1.0,1.186694184425838e-2,7.121205411175519e-5,
			    2.847142454085055e-7,8.364240962784899e-10,1.858128283833724e-12,
			    3.018846060781846e-15,3.015798735815980e-18};
const double Bessjy::y1pn[]={1.0,1.014039111045313,2.426762348629863e-1,
			     1.350308200342000e-2,9.516522033988099e-5};
const double Bessjy::y1pd[]={1.0,1.012208056357845,2.408580305488938e-1,
			     1.309511056184273e-2,7.746422941504713e-5};
const double Bessjy::y1qn[]={4.687499999999991e-2,5.652407388406023e-2,
			     1.676531273460512e-2,1.231216817715814e-3,1.178364381441801e-5};
const double Bessjy::y1qd[]={1.0,1.210119370463693,3.626494789275638e-1,
			     2.761695824829316e-2,3.240517192670181e-4};
const double Bessik::i0p[]={9.999999999999997e-1,2.466405579426905e-1,
			    1.478980363444585e-2,3.826993559940360e-4,5.395676869878828e-6,
			    4.700912200921704e-8,2.733894920915608e-10,1.115830108455192e-12,
			    3.301093025084127e-15,7.209167098020555e-18,1.166898488777214e-20,
			    1.378948246502109e-23,1.124884061857506e-26,5.498556929587117e-30};
const double Bessik::i0q[]={4.463598170691436e-1,1.702205745042606e-3,
			    2.792125684538934e-6,2.369902034785866e-9,8.965900179621208e-13};
const double Bessik::i0pp[]={1.192273748120670e-1,1.947452015979746e-1,
			     7.629241821600588e-2,8.474903580801549e-3,2.023821945835647e-4};
const double Bessik::i0qq[]={2.962898424533095e-1,4.866115913196384e-1,
			     1.938352806477617e-1,2.261671093400046e-2,6.450448095075585e-4,
			     1.529835782400450e-6};
const double Bessik::i1p[]={5.000000000000000e-1,6.090824836578078e-2,
			    2.407288574545340e-3,4.622311145544158e-5,5.161743818147913e-7,
			    3.712362374847555e-9,1.833983433811517e-11,6.493125133990706e-14,
			    1.693074927497696e-16,3.299609473102338e-19,4.813071975603122e-22,
			    5.164275442089090e-25,3.846870021788629e-28,1.712948291408736e-31};
const double Bessik::i1q[]={4.665973211630446e-1,1.677754477613006e-3,
			    2.583049634689725e-6,2.045930934253556e-9,7.166133240195285e-13};
const double Bessik::i1pp[]={1.286515211317124e-1,1.930915272916783e-1,
			     6.965689298161343e-2,7.345978783504595e-3,1.963602129240502e-4};
const double Bessik::i1qq[]={3.309385098860755e-1,4.878218424097628e-1,
			     1.663088501568696e-1,1.473541892809522e-2,1.964131438571051e-4,
			     -1.034524660214173e-6};
const double Bessik::k0pi[]={1.0,2.346487949187396e-1,1.187082088663404e-2,
			     2.150707366040937e-4,1.425433617130587e-6};
const double Bessik::k0qi[]={9.847324170755358e-1,1.518396076767770e-2,
			     8.362215678646257e-5};
const double Bessik::k0p[]={1.159315156584126e-1,2.770731240515333e-1,
			    2.066458134619875e-2,4.574734709978264e-4,3.454715527986737e-6};
const double Bessik::k0q[]={9.836249671709183e-1,1.627693622304549e-2,
			    9.809660603621949e-5};
const double Bessik::k0pp[]={1.253314137315499,1.475731032429900e1,
			     6.123767403223466e1,1.121012633939949e2,9.285288485892228e1,
			     3.198289277679660e1,3.595376024148513,6.160228690102976e-2};
const double Bessik::k0qq[]={1.0,1.189963006673403e1,5.027773590829784e1,
			     9.496513373427093e1,8.318077493230258e1,3.181399777449301e1,
			     4.443672926432041,1.408295601966600e-1};
const double Bessik::k1pi[]={0.5,5.598072040178741e-2,1.818666382168295e-3,
			     2.397509908859959e-5,1.239567816344855e-7};
const double Bessik::k1qi[]={9.870202601341150e-1,1.292092053534579e-2,
			     5.881933053917096e-5};
const double Bessik::k1p[]={-3.079657578292062e-1,-8.109417631822442e-2,
			    -3.477550948593604e-3,-5.385594871975406e-5,-3.110372465429008e-7};
const double Bessik::k1q[]={9.861813171751389e-1,1.375094061153160e-2,
			    6.774221332947002e-5};
const double Bessik::k1pp[]={1.253314137315502,1.457171340220454e1,
			     6.063161173098803e1,1.147386690867892e2,1.040442011439181e2,
			     4.356596656837691e1,7.265230396353690,3.144418558991021e-1};
const double Bessik::k1qq[]={1.0,1.125154514806458e1,4.427488496597630e1,
			     7.616113213117645e1,5.863377227890893e1,1.850303673841586e1,
			     1.857244676566022,2.538540887654872e-2};

