
#ifndef CPTCL_SCHEMA
#define CPTCL_SCHEMA

#include <string>

struct CPTCLSchema {

  // Elements of schema
  static std::string typeAtt;

  static std::string indexOrderAtt; //component major/minor, index C/Fortran; compMinorC is default

  static std::string numSpatialDimsAtt;
  static std::string numSpatialDimsAtt_deprecated;

  static std::string labelsAtt;
  static std::string labelsKey;

  static std::string velocityAtt;
  static std::string velocityKey;

  static std::string varKey;
  static std::string varsKey;
  static std::string varWithMeshKey;


  static std::string nodalCenteringKey;// Default
  static std::string edgeCenteringKey;
  static std::string faceCenteringKey;
  static std::string zonalCenteringKey;

  // Index ordering...
  static std::string compMajorCKey;
  static std::string compMinorCKey;
  static std::string compMajorFKey;
  static std::string compMinorFKey;


  //time
  static std::string timeKey;
  static std::string timeAtt;
  static std::string cycleAtt;
  
  //run info
  static std::string runInfoKey;
  static std::string softwareAtt;
};

#endif

