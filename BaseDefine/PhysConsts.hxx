// ----------------------------------------------------------------------
//
// File:	PhysConsts.hxx
//
// Purpose:	A holder for the physical constants
//              The value of the kbEVperK was taken from the information
//              available at the site:
//              http://physics.nist.gov/cuu/Constants/index.html 
// ----------------------------------------------------------------------

#ifndef _PhysConsts_HeaderFile
#define _PhysConsts_HeaderFile

// standard includes
#include <cmath>


/**
 * Class of objects that hold the physical constants.
 * Different instantiations could hold constants in different
 * physical units.  The constants here are all double's and
 * should be converted once before use by an object to the correct type.
 *
 * Current we have one global object, mksConsts, of this type.
 * It contains the values for the MKS/SI system.
 *
 */
struct PhysConsts {

/**
 * Create a set of physical constants
 *
 * @param ci the speed of light
 * @param eps0 free-space permittivity: the inverse of the factor in Poisson's equation, del * E = rho/eps0.
 * @param em electron mass
 * @param ec electron charge
 * @param mu0i free-space permiability
 * @param hi  Planck's constant
 * @param kbi Boltzmann's constant
 */ 
  PhysConsts( double ci, double eps0, double em, double ec, double mu0i, double hi, double kbi)
  {
    c = ci;
    h = hi;
    kb = kbi;
    epsilon0 = eps0;
    mu0 = mu0i;
    pi = acos(-1.);
    electronMass = em;
    electronCharge = ec;    
    electronChargeToMass = ec/em;
    // Secondary parameters
    twopi = 2.0*pi;
    fourpi = 4.0*pi;
    piover2 = pi/2.0;
    //mu0 = fourpi*0.0000001; This is only for MKS, so PHS removed on 08/04
    c2 = c*c;
    c2inv = 1.0/c2;
    c2oe = c2/eps0;
    coulombConst = 1./(fourpi*epsilon0);
    hbar = h/twopi;
  }

  /*** The speed of light */
  double c;
  
  /*** The planck constant */
  double h;
  double hbar;
  
  /*** The Boltzmann constant; the kbEVperK is in eV/K units. */
  double kb;
  double kbEVperK;
  
  /*** Free space permitivity, e.g. 1/4 pi k_1, where k_1 is the Coulomb force constant. */
  double epsilon0;
  
  /*** Permeablilty of free space */
  double mu0;

  /*** Pi */
  double pi;


  //
  // Beyond here are secondary parameters
  //
  
  /*** 2 times Pi */
  double twopi;
  
  /*** 4 times Pi */
  double fourpi;
  
  /*** Pi divided by two */
  double piover2;
  
  /*** The speed of light squared  */
  double c2;
  
  /*** The inverse of the speed of light squared */
  double c2inv;
  
  /*** The speed of light squared over epsilon_0  */
  double c2oe;
  
  /*** The Coulomb constant  */
  double coulombConst;
  
  /*** The mass of an electon */
  double electronMass;
  
  /*** The charge of an electron  */
  double electronCharge;
  
  /*** The charge-to-mass ratio of an electron  */
  double electronChargeToMass; 
};

/*** Holder for the constants in MKS */
extern PhysConsts mksConsts;

#endif
