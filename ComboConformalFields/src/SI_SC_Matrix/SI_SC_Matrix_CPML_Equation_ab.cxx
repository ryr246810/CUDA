#include "SI_SC_Matrix_CPML_Equation.hxx"
#include "PhysConsts.hxx"

void 
Matrix_Compute_a_b_SI_SC(DataBase *theData,
                         const Standard_Integer theTruncDir,
                         const Standard_Real Dt,
                         Standard_Real &a,
                         Standard_Real &b)
{
    Standard_Real Sigma = theData->GetPMLSigma(theTruncDir);
    Standard_Real Alpha = theData->GetPMLAlpha(theTruncDir);
    Standard_Real Kappa = theData->GetPMLKappa(theTruncDir);
    Standard_Real Epsilon = 0.0;

    Epsilon = theData->GetEpsilon();

    b = (Sigma + Kappa * Alpha) / Epsilon;
    a = Sigma / Epsilon;
}


void 
Matrix_Get_a_b_SI_SC(DataBase* theData, 
                     const Standard_Integer theTruncDir, 
                     const Standard_Real Dt, 
                     Standard_Real &a, 
                     Standard_Real &b)
{
  Standard_Real b_Paramt = theData->GetPML_b(theTruncDir);
  Standard_Real a_Paramt = theData->GetPML_a(theTruncDir);

  Standard_Real  Kappa = theData->GetPMLKappa(theTruncDir);

  b = Kappa / (Kappa + Dt * b_Paramt);
  a = -1.0 * Dt * a_Paramt / Kappa / (Kappa + Dt * b_Paramt);
}



void 
Matrix_Compute_Dual_a_b_SI_SC(DataBase* theData, 
                             const Standard_Integer theTruncDir, 
                             const Standard_Real Dt, 
                             Standard_Real &a, 
                             Standard_Real &b)
{
  Matrix_Compute_a_b_SI_SC(theData, theTruncDir, Dt, a, b);
}
