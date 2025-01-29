//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
/////////////////////////////////////////////////////
// Code by luca brombal, INFN -Trieste - 12.06.2009
/////////////////////////////////////////////////////

//
// $Id: PepiDetectorConstruction2.cc 71079 2013-06-10 20:37:11Z ihrivnac $
//
/// \file PepiDetectorConstruction2.cc
/// \brief Implementation of the PepiDetectorConstruction2 class

#include "PepiDetectorConstruction2.hh"
#include "PepiDetectorMessenger.hh"
//#include "PepiImageQualityPhantomParam.hh"

#include "G4RunManager.hh"

#include "G4NistManager.hh"
#include "G4MaterialPropertiesTable.hh"

#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Trd.hh"
#include "G4Sphere.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PVPlacement.hh"
#include "G4PVParameterised.hh"
#include "G4VPVParameterisation.hh"

#include "G4SDManager.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4VPrimitiveScorer.hh"
#include "G4PSEnergyDeposit.hh"
#include "G4PSFlatSurfaceCurrent.hh"
#include "G4PSCylinderSurfaceCurrent.hh"
#include "PepiPSPixiRad.hh"
//#include "PepiPSIoC.hh"
#include "G4SDParticleFilter.hh"
#include "G4SDParticleWithEnergyFilter.hh"

#include "G4VisAttributes.hh"
#include "G4UIcommand.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include <tuple>


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

//G4ThreadLocal G4bool PepiDetectorConstruction2::fConstructedSDandField = false;

namespace PEPI2
{

PepiDetectorConstruction2::PepiDetectorConstruction2()
: G4VUserDetectorConstruction(),
  fConstructed(false),
  fConstructedSDandField(false),
  fBidimensional(false),
  fWorldMaterial(0),
  fDetectorMaterial(0),
  fMaskMaterial(0),
  fObjectMaterial(0),
  fObject2Material(0),
  fObject3Material(0),
  fObject5Material(0),
  fObject6Material(0),
  fSubMaterial(0),
  fSphereMaterial(0),
  fMuscleMaterial(0),
  fWaxMaterial(0),
  fWorldSolid(0),
  fPixiRadSolid(0),
  fPixelSolid(0),
  fPixelSolid2(0),
  fM2subSolid(0),
  fM1subSolid(0),
  fM2Solid(0),
  fM1Solid(0),
  fEnvelopeM2Solid(0),
  fEnvelopeM1Solid(0),
  fObjectSolid(0),
  fObject2Solid(0),
  fObject3Solid(0),
  fObject5Solid(0),
  fObject6Solid(0),
  fSphereSolid(0),
  fSphere2Solid(0),
  fSandSolid(0),
  fSand2Solid(0),
  fSand3Solid(0),
  fSand4Solid(0),
  fSand5Solid(0),
  fSand6Solid(0),
  fSand7Solid(0),
  fSand8Solid(0),
  fSand9Solid(0),
  fSphere4Solid(0),
  fSphere5Solid(0),
  fSphere6Solid(0),
  fSphere7Solid(0),
  fSphere8Solid(0),
  fSphere9Solid(0),
  fSphere10Solid(0),
  fSphere11Solid(0),
  fSphere12Solid(0),
  fSphere13Solid(0),
  fSphere14Solid(0),
  fWorldLogical(0),
  fPixiRadLogical(0),
  fPixelLogical(0),
  fPixelLogical2(0),
  fM2Logical(0),
  fM1Logical(0),
  fM1subLogical(0),
  fM2subLogical(0),  
  fEnvelopeM2Logical(0),
  fEnvelopeM1Logical(0),
  fObjectLogical(0),
  fObject2Logical(0),
  fObject3Logical(0),
  fObject5Logical(0),
  fObject6Logical(0),
  fSphereLogical(0),
  fSphere2Logical(0),
  fSandLogical(0),
  fSand2Logical(0),
  fSphere4Logical(0),
  fSand3Logical(0),
  fSand4Logical(0),
  fSand5Logical(0),
  fSand6Logical(0),
  fSand7Logical(0),
  fSand8Logical(0),
  fSand9Logical(0),
  fSphere5Logical(0),
  fSphere6Logical(0),
  fSphere7Logical(0),
  fSphere8Logical(0),
  fSphere9Logical(0),
  fSphere10Logical(0),
  fSphere11Logical(0),
  fSphere12Logical(0),
  fSphere13Logical(0),
  fSphere14Logical(0),
  fScoringVolume(0),
  fWorldPhysical(0),
  fPixiRadPhysical(0),
  fPixelPhysical(0),
  fPixelPhysical2(0),
  fM2Physical(0),
  fM1Physical(0),
  fM1subPhysical(0),
  fM2subPhysical(0),  
  fEnvelopeM2Physical(0),
  fEnvelopeM1Physical(0),
  fObjectPhysical(0),
  fObject2Physical(0),
  fObject3Physical(0),
  fObject5Physical(0),
  fObject6Physical(0),
  fSpherePhysical(0),
  fSphere2Physical(0),
  fSandPhysical(0),
  fSand2Physical(0),
  fSphere4Physical(0),
  fSand3Physical(0),
  fSand4Physical(0),
  fSand5Physical(0),
  fSand6Physical(0),
  fSand7Physical(0),
  fSand8Physical(0),
  fSand9Physical(0),
  fSphere5Physical(0),
  fSphere6Physical(0),
  fSphere7Physical(0),
  fSphere8Physical(0),
  fSphere9Physical(0),
  fSphere10Physical(0),
  fSphere11Physical(0),
  fSphere12Physical(0),
  fSphere13Physical(0),
  fSphere14Physical(0),
  fRotAngle(0*deg),
  fTrans(0*um),
  fDith(0*um),
  fObjectDetDistance(10*cm),
  fSrcObjDistance(0.7*m),
  fObjSizeR(0),
  fObjSizeY(0),
  fObjSizeX(0),
  fStartAngle(0),
  fEndAngle(0),
  fWorldSizeX(0),
  fWorldSizeY(0),
  fWorldSizeZ(0),
  fOffset(50*cm),
  fnPixelsX(0),
  fnPixelsY(0),
  fPixiRadSizeX(0),
  fPixiRadSizeY(0),
  fPixiRadSizeZ(0),
  fPixiRadSize2X(0),
  fPixiRadSize2Y(0),
  fPixiRadSize2Z(0),
  fPixiRadSize3Z(0),
  fPixiRadSize4Z(0),
  fPixiRadSize3Y(0),
  fPixiRadSize3X(0),
  fInnerRadious(0),
  fOuterRadious(0),
  fLenghtTube(0),
  fMaskThickness(300*um),
  fM2Aperture(15*um),
  fM2Pitch(62*um),
  fSubThickness(525*um),
  fSourcePosZ(-85*cm),
  fThreshold1(3*keV),
  fThreshold2(3*keV),
  fAcquisitionType("doublemask"),
  fDetType("0COL"),
  fCheckOverlaps(false),
  fMessenger(0)
{
  // - All geometrical parameters depend on the object size
  // The object is defined as a full cylinder 
  // Inside the object there will be details of different 
  // materials 

  fWorldSizeX = 1*m;
  fWorldSizeY = 1*m;
  fWorldSizeZ = 2.3*m;

  fObjSizeR = 0.1*cm;
  fObjSizeY = 2*cm;  
  fObjSizeX = 2*cm;
  
  fXd=0.025*mm; 
  fXd2=0.025*mm; 
  
  fStartAngle = 0*deg;
  fEndAngle = 90*deg;

  fSkinThickness = 1.45*mm;

  fDetailSizeR = 2.5*cm;

  fPixelSizeX =  55*um;
  fPixelSizeY =  55*um;
  fPixelSizeZ =  1000*um;
  
  fPixelSize2X =  58.2*um;
  fPixelSize2Y =  58.2*um;
  fPixelSize2Z =  58.2*um;
  
  fPixelSize3X = 30*um;
  fPixelSize3Y = 30*um;
  fPixelSize3Z = 250*um;
  
  fTras = 28.3*cm;
  fTrasX = 240*um;
  fTrasY = 180*um;
  
  fMessenger = new PepiDetectorMessenger(this);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PepiDetectorConstruction2::~PepiDetectorConstruction2()
{ 
  delete fMessenger;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* PepiDetectorConstruction2::Construct()
{ 
  
    fConstructed = true;
    // - Define the Materials
    DefineMaterials();
    // - Define the Volumes
    DefineVolumes();
  

  return fWorldPhysical;
}

void PepiDetectorConstruction2::ConstructSDandField()
{
  
    fConstructedSDandField = true;
    // - Define the Sensitive Detectors
    DefineDetectors();
  }


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PepiDetectorConstruction2::DefineMaterials()
{ 
  G4NistManager* nist = G4NistManager::Instance();

  // ========================================
  //              MATERIALS
  // ========================================

  G4Material* CdTe          = nist->FindOrBuildMaterial("G4_CADMIUM_TELLURIDE");
  G4Material* Air           = nist->FindOrBuildMaterial("G4_AIR");
  G4Material* PlexiGlass    = nist->FindOrBuildMaterial("G4_PLEXIGLASS");
  G4Material* Water         = nist->FindOrBuildMaterial("G4_WATER");
//  G4Material* Nylon  	    = nist->FindOrBuildMaterial("G4_NYLON-6-6");
  G4Material* PolyCarbonate = nist->FindOrBuildMaterial("G4_POLYCARBONATE");
  G4Material* Silicon       = nist->FindOrBuildMaterial("G4_Si");
  G4Material* Graphite      = nist->FindOrBuildMaterial("G4_GRAPHITE");
  G4Material* Wax           = nist->FindOrBuildMaterial("G4_MIX_D_WAX"); 
  G4Material* Trioxide      = nist->FindOrBuildMaterial("G4_ALUMINUM_OXIDE"); 
  G4Material* Cellulose     = nist->FindOrBuildMaterial("G4_CELLULOSE_CELLOPHANE");
  G4Material* C             = nist->FindOrBuildMaterial("G4_C");
  G4Element* H              = nist->FindOrBuildElement("H");
  G4Element* P              = nist->FindOrBuildElement("P");
  G4Element* O              = nist->FindOrBuildElement("O");
  G4Element* Ca             = nist->FindOrBuildElement("Ca");
  
  
 // G4Material* Blood         = nist->FindOrBuildMaterial("G4_BLOOD_ICRP");
  
  

  // - Useful materials for TEST purposes...
  //G4Material* Vacuum1 = new G4Material("VACUUM1", 1., 1.01*g/mole, universe_mean_density); 
  //G4Material* Lead = nist->FindOrBuildMaterial("G4_Pb");
  G4Material* Gold = nist->FindOrBuildMaterial("G4_Au");

//=========================================
  //         SAND PAPER MATERIAL
  //=========================================
  
  G4double fractionmass, density1;
  G4String name;
  G4int ncomponents;
  
  density1 = 3.21*g/cm3;//3.21*g/cm3;
  G4Material* SiC = new G4Material(name="SandP",density1,ncomponents=2);
  SiC ->AddMaterial(Silicon, fractionmass=70.05*perCent);
  SiC ->AddMaterial(C, fractionmass=29.95*perCent);
  
  //=========================================
  //         HIDROXYAPATITE MATERIAL (HCa5P3O13)
  //=========================================
/*
  G4double density2; 
  
  density2 = 3.16*g/cm3;
  G4int natoms;
  
  G4Material* Hidrox = new G4Material(name="Hidrox2", density2, ncomponents=4);
  Hidrox ->AddElement(H, natoms=1);
  Hidrox ->AddElement(Ca, natoms=5);
  Hidrox ->AddElement(P, natoms=3);
  Hidrox ->AddElement(O, natoms=13);
*/  
 // density3 = 3.1*g/cm3;

 // G4Material* MicroC = new G4Material(name="MicroC", density3, ncomponents=2);
 // MicroC -> AddMaterial(Hidrox, fractionmass=70*perCent);
 // MicroC -> AddMaterial(Water, fractionmass=30*perCent);
 
 
   G4double zH,aH,zC,aC,zO,aO,zN,aN,zAr,aAr,zI,aI,zNa,aNa,zP,aP,zS,aS,zCl,aCl,zK,aK,zFe,aFe,zAl,aAl,aCa,zCa,aSi,zSi,fractionmassA,densityN, densityA,fractionmassPI, densityPI,fractionmassB, densityB, densityAl, densityHPA,densityFGR,densityDBT1, fractionmassFGR,densityADP,fractionmassADP,densityDBT,fractionmassDBT,fractionmassDBT1,densityPQ595,fractionmassPQ595,fractionmassPQ3070,densityPQ3070,fractionmassPQ5050,densityPQ5050,densitySilicone,densityPMMA,fractionmassPMMA,fractionmassMuscle,densityMuscle,fractionmassAorta,densityAorta;
  
  G4String nameNy, nameAll, symbolH,nameH, symbolC,nameC, symbolO,nameO, symbolN,nameN, symbolAr,nameAr, symbolI,nameI, symbolNa,nameNa, symbolP,nameP, symbolS,nameS, symbolCl,nameCl, symbolK,nameK, symbolFe,nameFe, symbolAl,nameSi,symbolSi,nameA, namePI, nameB, nameAl, nameHPA,nameCa,symbolCa,nameFGR,nameADP,nameDBT,nameDBT1,namePQ595,namePQ3070,namePQ5050,nameSilicone,namePMMA,nameMuscle,nameAorta;
  
  G4int ncomponentsNy,natomsNy, ncomponentsA, ncomponentsPI, ncomponentsB, ncomponentsAl, natomsAl, ncomponentsHPA, natomsHPA,natomsSilicone, ncomponentsFGR,ncomponentsADP,ncomponentsDBT,ncomponentsDBT1,ncomponentsPQ595,ncomponentsPQ3070,ncomponentsPQ5050,ncomponentsSilicone,ncomponentsPMMA,ncomponentsMuscle,ncomponentsAorta;

  aN = 14.01*g/mole;
  G4Element* Nitrogen1 = new G4Element(nameN="Nitrogen",symbolN="N" , zN= 7., aN);
  aO = 16.00*g/mole;
  G4Element* Oxygen1 = new G4Element(nameO="Oxygen" ,symbolO="O" , zO= 8., aO);
  aH = 1.00*g/mole;
  G4Element* Hydrogen1 = new G4Element(nameH="Hydrogen",symbolH="H" , zH= 1., aH);
  aC = 12.00*g/mole;
  G4Element* Carbon1 = new G4Element(nameC="Carbon" ,symbolC="C" , zC= 6., aC);
  aAr = 40.00*g/mole;
  G4Element* Argon1 = new G4Element(nameAr="Argon",symbolAr="Ar" , zAr= 18., aAr);
  aI = 127.00*g/mole;
  G4Element* Iodine1 = new G4Element(nameI="Iodine" ,symbolI="I" , zI= 53., aI);
  aNa = 23.00*g/mole;
  G4Element* Sodium1 = new G4Element(nameNa="Sodium",symbolNa="Na" , zNa= 11., aNa);
  aP = 31.00*g/mole;
  G4Element* Phosphorus1 = new G4Element(nameP="Phosphorus" ,symbolP="P" , zP= 15., aP);
  aS = 32.00*g/mole;
  G4Element* Sulfur1 = new G4Element(nameS="Sulfur",symbolS="S" , zS= 16., aS);
  aCl = 35.40*g/mole;
  G4Element* Chlorine1 = new G4Element(nameCl="Chlorine" ,symbolCl="Cl" , zCl= 17., aCl);
  aK = 39.10*g/mole;
  G4Element* Potassium1 = new G4Element(nameK="Potassium",symbolK="K" , zK= 19., aK);
  aFe = 55.90*g/mole;
  G4Element* Iron1 = new G4Element(nameFe="Iron" ,symbolFe="Fe" , zFe= 26., aFe);
  aAl = 27.00*g/mole;
  G4Element* Aluminium1 = new G4Element(nameAll="Aluminium" ,symbolAl="Al" , zAl= 13., aAl);
  aCa = 40.10*g/mole;
  G4Element* Calcium1 = new G4Element(nameCa="Calcium" ,symbolCa="Ca" , zCa= 20., aCa);
  aSi = 28.10*g/mole;
  G4Element* Silicon1 = new G4Element(nameSi="Silicon" ,symbolSi="Si" , zSi= 14., aSi);
//  aMo = 95.95*g/mole;
  G4Material* Molybdenum1 = nist->FindOrBuildMaterial("G4_Mo");
//  aRh = 102.91*g/mole;
  G4Material* Rhodium1 = nist->FindOrBuildMaterial("G4_Rh");
//  aAg = 107.86*g/mole;
  G4Material* Silver1 = nist->FindOrBuildMaterial("G4_Ag");
  //Nylon
  densityN = 3.4*g/cm3;
  G4Material* Nylon = new G4Material(nameNy="Nylon", densityN, ncomponentsNy = 4);
  Nylon -> AddElement(Carbon1, natomsNy = 6);
  Nylon -> AddElement(Hydrogen1, natomsNy = 11);
  Nylon -> AddElement(Nitrogen1, natomsNy = 1);
  Nylon -> AddElement(Oxygen1, natomsNy = 1);  


  //Air
  densityA = 0.0013*g/cm3;
  G4Material* AAir = new G4Material(nameA="AAir",densityA,ncomponentsA=4);
  AAir ->AddElement(Carbon1, fractionmassA=0.0097*perCent);
  AAir ->AddElement(Oxygen1, fractionmassA=23.1801*perCent);
  AAir ->AddElement(Nitrogen1, fractionmassA=75.5301*perCent);
  AAir ->AddElement(Argon1, fractionmassA=1.2801*perCent);
  
  //P.Iodine
  densityPI = 0.8*g/cm3;
  G4Material* PIodine = new G4Material(namePI="PIodine",densityPI,ncomponentsPI=5);
  PIodine ->AddElement(Carbon1, fractionmassPI=1.981*perCent);
  PIodine ->AddElement(Oxygen1, fractionmassPI=80.371*perCent);
  PIodine ->AddElement(Nitrogen1, fractionmassPI=0.381*perCent);
  PIodine ->AddElement(Hydrogen1, fractionmassPI=10.321*perCent);
  PIodine ->AddElement(Iodine1, fractionmassPI=6.946*perCent);
  
  //Blood
  densityB = 1.06*g/cm3;
  G4Material* Blood = new G4Material(nameB="Blood",densityB,ncomponentsB=10);
  Blood ->AddElement(Carbon1, fractionmassB=11*perCent);
  Blood ->AddElement(Oxygen1, fractionmassB=74.5*perCent);
  Blood ->AddElement(Nitrogen1, fractionmassB=3.3*perCent);
  Blood ->AddElement(Hydrogen1, fractionmassB=10.2*perCent);
  Blood ->AddElement(Sodium1, fractionmassB=0.1*perCent);
  Blood ->AddElement(Phosphorus1, fractionmassB=0.1*perCent);
  Blood ->AddElement(Sulfur1, fractionmassB=0.2*perCent);
  Blood ->AddElement(Chlorine1, fractionmassB=0.3*perCent);
  Blood ->AddElement(Potassium1, fractionmassB=0.2*perCent);
  Blood ->AddElement(Iron1, fractionmassB=0.1*perCent);
  
  //Alumina
  densityAl = 3.961*g/cm3;
  G4Material* Alumina = new G4Material(nameAl="Alumina",densityAl,ncomponentsAl=2);
  Alumina->AddElement(Aluminium1, natomsAl=2);
  Alumina->AddElement(Oxygen1, natomsAl=3);
  
  //HP
  densityHPA = 3.961*g/cm3;
  G4Material* Hydroxyapatite = new G4Material(nameHPA="Hydroxyapatite",densityHPA,ncomponentsHPA=4);
  Hydroxyapatite->AddElement(Calcium1, natomsHPA=10);
  Hydroxyapatite->AddElement(Oxygen1, natomsHPA=26);
  Hydroxyapatite->AddElement(Phosphorus1, natomsHPA=6);
  Hydroxyapatite->AddElement(Hydrogen1, natomsHPA=2);
  
  //Fibroglandular
  densityFGR = 1.020*g/cm3;
  G4Material*  FibroGland= new G4Material(nameFGR="Fibroglandular",densityFGR,ncomponentsFGR=8);
  FibroGland->AddElement(Carbon1, fractionmassFGR=33.2*perCent);
  FibroGland->AddElement(Oxygen1, fractionmassFGR=52.7*perCent);
  FibroGland->AddElement(Phosphorus1, fractionmassFGR=0.1*perCent);
  FibroGland->AddElement(Hydrogen1, fractionmassFGR=10.6*perCent);
  FibroGland->AddElement(Nitrogen1, fractionmassFGR=3.0*perCent);
  FibroGland->AddElement(Sodium1, fractionmassFGR=0.1*perCent);
  FibroGland->AddElement(Sulfur1, fractionmassFGR=0.2*perCent);
  FibroGland->AddElement(Chlorine1, fractionmassFGR=0.1*perCent);
  
  //Aorta
  densityAorta = 1.05*g/cm3;
  G4Material*  Aorta= new G4Material(nameAorta="Aorta",densityAorta,ncomponentsAorta=9);
  Aorta->AddElement(Carbon1, fractionmassAorta=14.7*perCent);
  Aorta->AddElement(Oxygen1, fractionmassAorta=69.8*perCent);
  Aorta->AddElement(Phosphorus1, fractionmassAorta=0.4*perCent);
  Aorta->AddElement(Hydrogen1, fractionmassAorta=9.9*perCent);
  Aorta->AddElement(Nitrogen1, fractionmassAorta=4.2*perCent);
  Aorta->AddElement(Sodium1, fractionmassAorta=0.2*perCent);
  Aorta->AddElement(Sulfur1, fractionmassAorta=0.3*perCent);
  Aorta->AddElement(Potassium1, fractionmassAorta=0.1*perCent);
  Aorta->AddElement(Calcium1, fractionmassAorta=0.4*perCent);
  
  //Adipose
  densityADP = 0.92*g/cm3;
  G4Material*  Adipose= new G4Material(nameADP="Adipose",densityADP,ncomponentsADP=6);
  Adipose->AddElement(Carbon1, fractionmassADP=64.0*perCent);
  Adipose->AddElement(Oxygen1, fractionmassADP=22.9*perCent);
  Adipose->AddElement(Phosphorus1, fractionmassADP=0.2*perCent);
  Adipose->AddElement(Hydrogen1, fractionmassADP=12.0*perCent);
  Adipose->AddElement(Nitrogen1, fractionmassADP=0.8*perCent);
  Adipose->AddElement(Calcium1, fractionmassADP=0.1*perCent);
  
   //PMMA
  densityPMMA = 1.20*g/cm3;
  G4Material*  Methacrylate= new G4Material(namePMMA="PMMA",densityPMMA,ncomponentsPMMA=3);
  Methacrylate->AddElement(Carbon1, fractionmassPMMA=59.9846*perCent);
  Methacrylate->AddElement(Oxygen1, fractionmassPMMA=31.9613*perCent);
  Methacrylate->AddElement(Hydrogen1, fractionmassPMMA=8.0541*perCent);
  
  //Muscle
  densityMuscle = 1.05*g/cm3;
  G4Material*  Muscle= new G4Material(nameMuscle="Muscle",densityMuscle,ncomponentsMuscle=9);
  Muscle->AddElement(Hydrogen1, fractionmassMuscle=10.2*perCent);
  Muscle->AddElement(Carbon1, fractionmassMuscle=14.3*perCent);
  Muscle->AddElement(Nitrogen1, fractionmassMuscle=3.4*perCent);
  Muscle->AddElement(Oxygen1, fractionmassMuscle=71.0*perCent);
  Muscle->AddElement(Sodium1, fractionmassMuscle=0.1*perCent);
  Muscle->AddElement(Phosphorus1, fractionmassMuscle=0.2*perCent);
  Muscle->AddElement(Sulfur1, fractionmassMuscle=0.3*perCent);
  Muscle->AddElement(Chlorine1, fractionmassMuscle=0.1*perCent);
  Muscle->AddElement(Potassium1, fractionmassMuscle=0.4*perCent);
  
  //Polydimethylsiloxane 
  densitySilicone = 0.965*g/cm3;
  G4Material* Silicone = new G4Material(nameSilicone="Silicone",densitySilicone,ncomponentsSilicone=4);
  Silicone->AddElement(Carbon1, natomsSilicone=2);
  Silicone->AddElement(Oxygen1, natomsSilicone=1);
  Silicone->AddElement(Silicon1, natomsSilicone=1);
  Silicone->AddElement(Hydrogen1, natomsSilicone=6);
  
  //DenseBreast
  densityDBT = 1.005*g/cm3;
  G4Material* DenseBreast = new G4Material(nameDBT="DenseBreast",densityDBT,ncomponentsDBT=2);
  DenseBreast ->AddMaterial(Adipose, fractionmassDBT=15*perCent);
  DenseBreast ->AddMaterial(FibroGland, fractionmassDBT=85*perCent);
  
  //StandardBreast
  densityDBT1 = 0.97*g/cm3;
  G4Material* StandardBreast = new G4Material(nameDBT1="StandardBreast",densityDBT1,ncomponentsDBT1=2);
  StandardBreast ->AddMaterial(Adipose, fractionmassDBT1=50*perCent);
  StandardBreast ->AddMaterial(FibroGland, fractionmassDBT1=50*perCent);
  
  //Plaque5-95
  densityPQ595 = 2.719*g/cm3;
  G4Material* PlaqueS = new G4Material(namePQ595="PlaqueS",densityPQ595,ncomponentsPQ595=2);
  PlaqueS ->AddMaterial(Hydroxyapatite, fractionmassPQ595=15*perCent);
  PlaqueS ->AddMaterial(Methacrylate, fractionmassPQ595=85*perCent);
  
  //Plaque30-70
  densityPQ3070 = 2.9383*g/cm3;
  G4Material* PlaqueI = new G4Material(namePQ3070="PlaqueI",densityPQ3070,ncomponentsPQ3070=2);
  PlaqueI ->AddMaterial(Hydroxyapatite, fractionmassPQ3070=30*perCent);
  PlaqueI ->AddMaterial(Methacrylate, fractionmassPQ3070=70*perCent);
  
  //Plaque50-50
  densityPQ5050 = 3.2305*g/cm3;
  G4Material* PlaqueC = new G4Material(namePQ5050="PlaqueI",densityPQ5050,ncomponentsPQ5050=2);
  PlaqueC ->AddMaterial(Hydroxyapatite, fractionmassPQ5050=50*perCent);
  PlaqueC ->AddMaterial(Methacrylate, fractionmassPQ5050=50*perCent);
  // ========================================
  //         REFRACTION COEFFICIENTS
  // ========================================

  
  // Load delta coefficients and the respective energy interval
  // Values taken from: http://ts-imaging.science.unimelb.edu.au/Services/Simple/ICUtilXdata.aspx
  std::vector<double> GraphiteDelta = LoadDelta("../data/Graphite_delta.txt");
  std::vector<double> SiliconDelta = LoadDelta("../data/Silicon_delta.txt");
  std::vector<double> PlexiGlassDelta = LoadDelta("../data/PMMA_delta.txt");
  std::vector<double> NylonDelta = LoadDelta("../data/Nylon_delta.txt");
  std::vector<double> PolyCarbonateDelta = LoadDelta("../data/Polycharbonate_delta.txt");
  std::vector<double> WaterDelta = LoadDelta("../data/Water_delta.txt");
  std::vector<double> energies = LoadDelta("../data/Energy.txt");
  std::vector<double> AAirDelta = LoadDelta("../data/Air_delta.txt");
  std::vector<double> PIodineDelta = LoadDelta("../data/P.Iodine_delta.txt");
  std::vector<double> BloodDelta = LoadDelta("../data/Blood_delta.txt");
  std::vector<double> AluminaDelta = LoadDelta("../data/Alumina_delta.txt");
  std::vector<double> HydroxyapatiteDelta = LoadDelta("../data/HA_delta.txt");
  std::vector<double> AortaDelta = LoadDelta("../data/Aorta_delta.txt");
  std::vector<double> BreastDelta = LoadDelta("../data/Wax_delta.txt");
  std::vector<double> GlandularDelta = LoadDelta("../data/Glandular_delta.txt");
  std::vector<double> MuscleDelta = LoadDelta("../data/Muscle_delta.txt");
  std::vector<double> SiliconeDelta = LoadDelta("../data/Silicone_delta.txt");
  std::vector<double> TrioxideDelta = LoadDelta("../data/Trioxide_Delta.txt");
  std::vector<double> CelluloseDelta = LoadDelta("../data/Cellulose_delta.txt");
  std::vector<double> SiCDelta = LoadDelta("../data/SiC_delta.txt");
  std::vector<double> WaxDelta = LoadDelta("../data/Wax_delta.txt");	
  
  
  G4int NumEntries = static_cast<int>(energies.size()); 
  std::vector<double> GraphiteRindex(NumEntries); 
  std::vector<double> SiliconRindex(NumEntries);
  std::vector<double> PlexiGlassRindex(NumEntries);
  std::vector<double> NylonRindex(NumEntries);
  std::vector<double> PolyCarbonateRindex(NumEntries);
  std::vector<double> WaterRindex(NumEntries);
  std::vector<double> CdTeRindex(NumEntries);
  std::vector<double> AirRindex(NumEntries);
  std::vector<double> GoldRindex(NumEntries);
  std::vector<double> AAirRindex(NumEntries);
  std::vector<double> PIodineRindex(NumEntries);
  std::vector<double> BloodRindex(NumEntries);
  std::vector<double> AluminaRindex(NumEntries);
  std::vector<double> HARindex(NumEntries);
  std::vector<double> SiliconeRindex(NumEntries);
  std::vector<double> MolybdenumRindex(NumEntries);
  std::vector<double> RhodiumRindex(NumEntries);
  std::vector<double> SilverRindex(NumEntries);
  std::vector<double> AortaRindex(NumEntries);
  std::vector<double> BreastRindex(NumEntries);
  std::vector<double> GlandularRindex(NumEntries);
  std::vector<double> MuscleRindex(NumEntries);
  std::vector<double> TrioxideRindex(NumEntries);
  std::vector<double> CelluloseRindex(NumEntries);
  std::vector<double> SiCRindex(NumEntries);
  std::vector<double> WaxRindex(NumEntries);
  
  for (G4int i = 0; i < NumEntries; ++i)
  {
    PlexiGlassRindex[i]     = 1 - PlexiGlassDelta[i];
    GraphiteRindex[i]       = 1 - GraphiteDelta[i];
    SiliconRindex[i]        = 1; //- SiliconDelta[i];

    WaterRindex[i]          = 1 - WaterDelta[i];
    NylonRindex[i]	    = 1 - NylonDelta[i];
    PolyCarbonateRindex[i]  = 1 - PolyCarbonateDelta[i];
    energies[i] = energies[i]*keV;
    CdTeRindex[i]           = 1;
    AirRindex[i]            = 1;
    GoldRindex[i]           = 1;
    AAirRindex[i]          = 1 - AAirDelta[i];
    PIodineRindex[i]	    = 1 - PIodineDelta[i];
    BloodRindex[i]  = 1 - BloodDelta[i];
    AluminaRindex[i]          = 1 - AluminaDelta[i];
    HARindex[i]	    = 1 - HydroxyapatiteDelta[i];
    SiliconeRindex[i]	    = 1 - SiliconeDelta[i];
    MolybdenumRindex[i]	    = 1;
    RhodiumRindex[i]	    = 1;
    SilverRindex[i]	    = 1;
    AortaRindex[i]	    = 1 - AortaDelta[i];
    BreastRindex[i]	    = 1 - BreastDelta[i];
    GlandularRindex[i]	    = 1 - GlandularDelta[i];
    MuscleRindex[i]	    = 1 - MuscleDelta[i];
    TrioxideRindex[i]       =1 - TrioxideDelta[i];
    CelluloseRindex[i]       = 1 - CelluloseDelta[i];
    SiCRindex[i]         = 1 - SiCDelta[i];
    WaxRindex[i]         = 1 - WaxDelta[i];
 }
 

  G4MaterialPropertiesTable* AAirMatPropTbl = new G4MaterialPropertiesTable();
  AAirMatPropTbl->AddProperty("RINDEX",energies.data(),AAirRindex.data(),NumEntries);
  AAir->SetMaterialPropertiesTable(AAirMatPropTbl);
  
  G4MaterialPropertiesTable* PIodineMatPropTbl = new G4MaterialPropertiesTable();
  PIodineMatPropTbl->AddProperty("RINDEX",energies.data(),PIodineRindex.data(),NumEntries);
  PIodine->SetMaterialPropertiesTable(PIodineMatPropTbl);
  
  G4MaterialPropertiesTable* BloodMatPropTbl = new G4MaterialPropertiesTable();
  BloodMatPropTbl->AddProperty("RINDEX",energies.data(),BloodRindex.data(),NumEntries);
  Blood->SetMaterialPropertiesTable(BloodMatPropTbl);
  
  G4MaterialPropertiesTable* AluminaMatPropTbl = new G4MaterialPropertiesTable();
  AluminaMatPropTbl->AddProperty("RINDEX",energies.data(),AluminaRindex.data(),NumEntries);
  Alumina->SetMaterialPropertiesTable(AluminaMatPropTbl);
  
  G4MaterialPropertiesTable* HAMatPropTbl = new G4MaterialPropertiesTable();
  HAMatPropTbl->AddProperty("RINDEX",energies.data(),HARindex.data(),NumEntries);
  Hydroxyapatite->SetMaterialPropertiesTable(HAMatPropTbl);
  
  G4MaterialPropertiesTable* DBTMatPropTbl = new G4MaterialPropertiesTable();
  DBTMatPropTbl->AddProperty("RINDEX",energies.data(),GlandularRindex.data(),NumEntries);
  DenseBreast->SetMaterialPropertiesTable(DBTMatPropTbl); 
  
  G4MaterialPropertiesTable* DBT1MatPropTbl = new G4MaterialPropertiesTable();
  DBT1MatPropTbl->AddProperty("RINDEX",energies.data(),BreastRindex.data(),NumEntries);
  StandardBreast->SetMaterialPropertiesTable(DBT1MatPropTbl);
  
  G4MaterialPropertiesTable* AortaMatPropTbl = new G4MaterialPropertiesTable();
  AortaMatPropTbl->AddProperty("RINDEX",energies.data(),AortaRindex.data(),NumEntries);
  Aorta->SetMaterialPropertiesTable(AortaMatPropTbl);
  
  
  G4MaterialPropertiesTable* SiliconeMatPropTbl = new G4MaterialPropertiesTable();
  SiliconeMatPropTbl->AddProperty("RINDEX",energies.data(),SiliconeRindex.data(),NumEntries);
  Silicone->SetMaterialPropertiesTable(SiliconeMatPropTbl);
  
  G4MaterialPropertiesTable* PlSMatPropTbl = new G4MaterialPropertiesTable();
  PlSMatPropTbl->AddProperty("RINDEX",energies.data(),PlexiGlassRindex.data(),NumEntries);
  PlaqueS->SetMaterialPropertiesTable(PlSMatPropTbl);
  
  G4MaterialPropertiesTable* PlIMatPropTbl = new G4MaterialPropertiesTable();
  PlIMatPropTbl->AddProperty("RINDEX",energies.data(),PlexiGlassRindex.data(),NumEntries);
  PlaqueI->SetMaterialPropertiesTable(PlIMatPropTbl);
  
  G4MaterialPropertiesTable* PlCMatPropTbl = new G4MaterialPropertiesTable();
  PlCMatPropTbl->AddProperty("RINDEX",energies.data(),PlexiGlassRindex.data(),NumEntries);
  PlaqueC->SetMaterialPropertiesTable(PlCMatPropTbl); 

  G4MaterialPropertiesTable* GraphiteMatPropTbl = new G4MaterialPropertiesTable();
  GraphiteMatPropTbl->AddProperty("RINDEX",energies.data(),GraphiteRindex.data(),NumEntries);
  Graphite->SetMaterialPropertiesTable(GraphiteMatPropTbl);  

  G4MaterialPropertiesTable* SiliconMatPropTbl = new G4MaterialPropertiesTable();
  SiliconMatPropTbl->AddProperty("RINDEX",energies.data(),SiliconRindex.data(),NumEntries);
  Silicon->SetMaterialPropertiesTable(SiliconMatPropTbl);  

  G4MaterialPropertiesTable* WaterMatPropTbl = new G4MaterialPropertiesTable();
  WaterMatPropTbl->AddProperty("RINDEX",energies.data(),WaterRindex.data(),NumEntries);
  Water->SetMaterialPropertiesTable(WaterMatPropTbl);    

  G4MaterialPropertiesTable* PlexiGlassMatPropTbl = new G4MaterialPropertiesTable();
  PlexiGlassMatPropTbl->AddProperty("RINDEX",energies.data(),PlexiGlassRindex.data(),NumEntries);
  PlexiGlass->SetMaterialPropertiesTable(PlexiGlassMatPropTbl);

  G4MaterialPropertiesTable* NylonMatPropTbl = new G4MaterialPropertiesTable();
  NylonMatPropTbl->AddProperty("RINDEX",energies.data(),NylonRindex.data(),NumEntries);
  Nylon->SetMaterialPropertiesTable(NylonMatPropTbl);

  G4MaterialPropertiesTable* PolyCarbonateMatPropTbl = new G4MaterialPropertiesTable();
  PolyCarbonateMatPropTbl->AddProperty("RINDEX",energies.data(),PolyCarbonateRindex.data(),NumEntries);
  PolyCarbonate->SetMaterialPropertiesTable(PolyCarbonateMatPropTbl);

  G4MaterialPropertiesTable* CdTeMatPropTbl = new G4MaterialPropertiesTable();
  CdTeMatPropTbl->AddProperty("RINDEX",energies.data(),CdTeRindex.data(),NumEntries);
  CdTe->SetMaterialPropertiesTable(CdTeMatPropTbl);

  G4MaterialPropertiesTable* AirMatPropTbl = new G4MaterialPropertiesTable();
  AirMatPropTbl->AddProperty("RINDEX",energies.data(),AirRindex.data(),NumEntries);
  Air->SetMaterialPropertiesTable(AirMatPropTbl);

  G4MaterialPropertiesTable* GoldMatPropTbl = new G4MaterialPropertiesTable();
  GoldMatPropTbl->AddProperty("RINDEX",energies.data(),GoldRindex.data(),NumEntries);
  Gold->SetMaterialPropertiesTable(GoldMatPropTbl);
  
  G4MaterialPropertiesTable* MoMatPropTbl = new G4MaterialPropertiesTable();
  MoMatPropTbl->AddProperty("RINDEX",energies.data(),MolybdenumRindex.data(),NumEntries);
  Molybdenum1->SetMaterialPropertiesTable(MoMatPropTbl);
  
  G4MaterialPropertiesTable* RhMatPropTbl = new G4MaterialPropertiesTable();
  RhMatPropTbl->AddProperty("RINDEX",energies.data(),RhodiumRindex.data(),NumEntries);
  Rhodium1->SetMaterialPropertiesTable(RhMatPropTbl);
  
  G4MaterialPropertiesTable* AgMatPropTbl = new G4MaterialPropertiesTable();
  AgMatPropTbl->AddProperty("RINDEX",energies.data(),SilverRindex.data(),NumEntries);
  Silver1->SetMaterialPropertiesTable(AgMatPropTbl);
  
  G4MaterialPropertiesTable* PMMAMatPropTbl = new G4MaterialPropertiesTable();
  PMMAMatPropTbl->AddProperty("RINDEX",energies.data(),PlexiGlassRindex.data(),NumEntries);
  Methacrylate->SetMaterialPropertiesTable(PMMAMatPropTbl);
  
  G4MaterialPropertiesTable* MuscleMatPropTbl = new G4MaterialPropertiesTable();
  MuscleMatPropTbl->AddProperty("RINDEX",energies.data(),MuscleRindex.data(),NumEntries);
  Muscle->SetMaterialPropertiesTable(MuscleMatPropTbl); 
  
  G4MaterialPropertiesTable* FibroGlandMatPropTbl = new G4MaterialPropertiesTable();
  FibroGlandMatPropTbl->AddProperty("RINDEX",energies.data(),NylonRindex.data(),NumEntries);
  FibroGland->SetMaterialPropertiesTable(FibroGlandMatPropTbl); 
  
  G4MaterialPropertiesTable* TrioxideMatPropTbl = new G4MaterialPropertiesTable();
  TrioxideMatPropTbl->AddProperty("RINDEX",energies.data(),TrioxideRindex.data(),NumEntries);
  Trioxide->SetMaterialPropertiesTable(TrioxideMatPropTbl);
  
  G4MaterialPropertiesTable* CelluloseMatPropTbl = new G4MaterialPropertiesTable();
  CelluloseMatPropTbl->AddProperty("RINDEX",energies.data(),CelluloseRindex.data(),NumEntries);
  Cellulose->SetMaterialPropertiesTable(CelluloseMatPropTbl);
  
  G4MaterialPropertiesTable* SiCMatPropTbl = new G4MaterialPropertiesTable();
  SiCMatPropTbl->AddProperty("RINDEX",energies.data(),SiCRindex.data(),NumEntries);
  SiC->SetMaterialPropertiesTable(SiCMatPropTbl); 
  
  G4MaterialPropertiesTable* WaxMatPropTbl = new G4MaterialPropertiesTable();
  WaxMatPropTbl->AddProperty("RINDEX",energies.data(),WaxRindex.data(),NumEntries);
  Wax->SetMaterialPropertiesTable(WaxMatPropTbl);


  
 
  
  // ========================================
  //           DEFAULT MATERIALS
  // ========================================

  fWorldMaterial    = Air;// Vacuum1;
  fIonCMaterial     = Air;// Vacuum1;
  fDetectorMaterial = CdTe;
  fMaskMaterial     = Gold;
  fObjectMaterial   = Air;//PlexiGlass;
  fObject2Material  = Air;//Wax;
  fObject3Material  = Air;//Wax;
  fSubMaterial	    = Air;//Wax;//Trioxide;
  fSphereMaterial   = Air;//SiC;
  fMuscleMaterial   = Air;//Cellulose;
  fObject5Material  = Blood;//
//  fObject6Material  = Molybdenum1; Silver1;
  fObject6Material  = PlaqueS;// Molybdenum1;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PepiDetectorConstruction2::DefineVolumes()
{
  
  // - PIXIRAD Parameters


  fnPixelsX = 256;

  // - The y-dimension of PixiRad changes if we want
  // a bidimensional acquisition or not
  if (fBidimensional)
  {
    fnPixelsY = 256;
  }
  else
  {
    fnPixelsY = 1;
  }

  fPixiRadSizeX = fPixelSizeX * fnPixelsX;
  fPixiRadSizeY = fPixelSizeY * fnPixelsY;
  fPixiRadSizeZ = fPixelSizeZ;
  
  fPixiRadSize3X = 10.8*cm;
  fPixiRadSize3Y = 10.2*cm;
  fPixiRadSize3Z = 3.375*cm;
  
  fPixiRadSize2X = fPixelSize2X * fnPixelsX;
  fPixiRadSize2Y = fPixelSize2Y * fnPixelsX;
  fPixiRadSize2Z = fPixelSize2X;
  
 // fPixiRadSize3Y = fPixelSize2X * fnPixelsX;
  
 // fPixiRadSize3Z = fPixelSize2X;
  
  fPixiRadSize4Z = fPixelSize3Z;

  // ========================================
  //                  WORLD
  // ========================================
  
  // - Build the WORLD as an unrotated Box in (0,0,0)
  fWorldSolid = new G4Box("World",                         //its name
                          fWorldSizeX/2,                   //its size
                          fWorldSizeY/2,
                          fWorldSizeZ/2);          
   
  fWorldLogical = new G4LogicalVolume(fWorldSolid,         //its solid
                                      fWorldMaterial,      //its material
                                      "World");            //its name
                       
  fWorldPhysical =  new G4PVPlacement(0,                   //no rotation
                                      G4ThreeVector(),     //at (0,0,0)
                                      fWorldLogical,       //its logical volume
                                      "World",             //its name
                                      0,                   //its mother  volume
                                      false,               //no boolean operation
                                      0,                   //copy number
                                      fCheckOverlaps);     //checking overlaps 


  // ========================================
  //                DETECTOR
  // ========================================

  // - Build the DETECTOR PIXEL UNIT as a Box
  fPixelSolid = new G4Box("Pixel",                          //its name
                          fPixelSizeX/2,                    //its size
                          fPixelSizeY/2,
                          fPixelSizeZ/2);
                     
  fPixelLogical = new G4LogicalVolume(fPixelSolid,          //its solid
                                      fDetectorMaterial,    //its material
                                      "PixelLV");           //its name

  // Build the Detector Envelope 
  fPixiRadSolid =  new G4Box("PixiRad",                    //its name                 
                             fPixiRadSizeX/2,              //its size
                             fPixiRadSizeY/2,
                             fPixiRadSizeZ/2);        
      
  fPixiRadLogical =  new G4LogicalVolume(fPixiRadSolid,    //its solid
                                         fWorldMaterial,   //its material
                                         "PixiRad");       //its name
                    
  // - Place the physical copies of the pixel in a x-y matrix
  // The full detector is built from the top-left corner in
  // [*][*][*][*][*][*][*][*][*][*][*][*] #1 row (fnPixelsX long)
  // [*][*][*][*]........................ #2 row (fnPixelsX long)
  // ....................................
  // .................................... # fnPixelsX * fnPixels Y
  G4int copy_no=0;  

  for (G4int iY = 0; iY < fnPixelsY ; iY++)
  {
    for (G4int iX = 0; iX < fnPixelsX ; iX++)
    {
      G4double x = + iX*fPixelSizeX - fPixiRadSizeX/2;// + fPixelSizeX/2;
      G4double y = - iY*fPixelSizeY + fPixiRadSizeY/2;// - fPixelSizeY/2;
      
      G4ThreeVector position = G4ThreeVector(x, y, 0);
      G4String  name = "Pixel_" + G4UIcommand::ConvertToString(copy_no);

      fPixelPhysical =  new G4PVPlacement(0,                           //its rotation
                                          position,                    //its position
                                          fPixelLogical,               //its logical volume
                                          name,                        //its name
                                          fPixiRadLogical,             //its mother volume
                                          false,                       //no boolean operation
                                          copy_no,                     //copy number
                                          fCheckOverlaps);             //checking overlaps 

      copy_no++;                              
    }
  }

  // - Place the Detector Envelope in the World
  G4ThreeVector positionPixirad = G4ThreeVector(0, 0, fSourcePosZ+fSrcObjDistance+fObjectDetDistance);
  fPixiRadPhysical = new G4PVPlacement(0,                                                  //its rotation
                                       positionPixirad,					   //its position
                                       fPixiRadLogical,                                    //its logical volume
                                       "PixiRad",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                       fCheckOverlaps);                                    //checking overlaps


  G4Region* aRegion = new G4Region("PixiRad");
  fPixiRadLogical->SetRegion(aRegion);
  aRegion->AddRootLogicalVolume(fPixiRadLogical);
  
 
       // ========================================
    //               Sand_paper 1
    // =========================================

 
                                 
 fSphereSolid = new G4Sphere("sphere1",
			      0*mm,                         //its R_min
			      100*um/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphereLogical = new G4LogicalVolume(fSphereSolid, 
				      fSphereMaterial,
				      "SphereLV");                                     
 
 
 
 
 // Build the Speckel's Envelope 
  fSandSolid =  new G4Box("SandDet",                    //its name                 
                             100*256*um/2,              //its size
                             100*256*um/2,
                             100*um/2);        
      
  fSandLogical =  new G4LogicalVolume(fSandSolid,    //its solid
                                      fWorldMaterial,   //its material
                                      "SandDet");       //its name
 
  
 // G4double x1 = + j*fPixelSize2X - fPixiRadSize2X/2;// + fPixelSizeX/2;
 //      G4double y1 = - i*fPixelSize2X + fPixiRadSize2X/2;// - fPixelSizeY/2;
 
  std::vector<G4ThreeVector> coordinates = ReadCoordinates("../data/100_um_Coord1.txt");
 G4int Numb_speck = 180000;
 
 for(G4int i = 0; i < Numb_speck; i ++)
 {
 	
 	  //G4double x1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2; 
 	  //G4double y1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2;
          //G4double z1 = G4UniformRand()*fPixiRadSize2Z - fPixiRadSize2Z/2;
      
      G4ThreeVector position1 = coordinates[i];// G4ThreeVector(x1, y1, 0);
      //G4ThreeVector position1 =  G4ThreeVector(x1, y1, 0);
      //G4String  names = "Speckle_" + G4UIcommand::ConvertToString(copy_nn);
      
      		fSpherePhysical = new G4PVPlacement(0, 
      						     position1 ,//+ G4ThreeVector(0*um, 0*um, fSourcePosZ+fSrcObjDistance -fTras -0.2*cm),
      					             fSphereLogical,
      					             "spheresplacement", 
      					             fSandLogical,
      					             false, 
      					             i, 
      						     fCheckOverlaps); 
       
	//copy_nn++;   
 	 
 }                             
 
 // - Place the Sandpaper Envelope in the World
  G4ThreeVector positionSand = G4ThreeVector(0*um+fTrasX, 0*um+fTrasY, fSourcePosZ+fSrcObjDistance -fTras);
  fSandPhysical = new G4PVPlacement(0,                                                  //its rotation
                                       positionSand,					   //its position
                                       fSandLogical,                                    //its logical volume
                                       "SandDet",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                       fCheckOverlaps);                                    //checking overlaps


  G4Region* aRegion2 = new G4Region("SandDet");
  fSandLogical->SetRegion(aRegion2);
  aRegion2->AddRootLogicalVolume(fSandLogical);
  
 
    // ========================================
    //               Sand_paper 2
    // ========================================
    

 
                                 
 fSphere5Solid = new G4Sphere("sphere5",
			      0*mm,                         //its R_min
			      125*um/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere5Logical = new G4LogicalVolume(fSphere5Solid, 
				      fSphereMaterial,
				      "SphereLV5");                                     
 
 
 
 
 // Build the Speckel's Envelope 
  fSand3Solid =  new G4Box("SandDet3",                    //its name                 
                            125*256*um/2,              //its size
                             125*256*um/2,
                             125*um/2);        
      
  fSand3Logical =  new G4LogicalVolume(fSand3Solid,    //its solid
                                      fWorldMaterial,   //its material
                                      "SandDet3");       //its name
 
  
  
  std::vector<G4ThreeVector> coordinates2 = ReadCoordinates("../data/125_um_Coord2.txt");
 G4int Numb_speck2 = 180000;
 
 for(G4int i = 0; i < Numb_speck2; i ++)
 {
 	
 	  //G4double x1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2; 
 	  //G4double y1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2;
          //G4double z1 = G4UniformRand()*fPixiRadSize2Z - fPixiRadSize2Z/2;
      
      G4ThreeVector position2 = coordinates2[i];// G4ThreeVector(x1, y1, 0);
      //G4ThreeVector position1 =  G4ThreeVector(x1, y1, 0);
      //G4String  names = "Speckle_" + G4UIcommand::ConvertToString(copy_nn);
      
      		fSphere5Physical = new G4PVPlacement(0, 
      						     position2 ,//+ G4ThreeVector(0*um, 0*um, fSourcePosZ+fSrcObjDistance -fTras -0.2*cm),
      					             fSphere5Logical,
      					             "spheresplacement", 
      					             fSand3Logical,
      					             false, 
      					             i, 
      						     fCheckOverlaps); 
       
	//copy_nn++;   
 	 
 }                                          
 
 // - Place the Sandpaper Envelope in the World
  G4ThreeVector positionSand3 = G4ThreeVector(0*um+fTrasX, 0*um+fTrasY, fSourcePosZ+fSrcObjDistance -fTras + fPixiRadSize2Z);
  fSand3Physical = new G4PVPlacement(0,                                                  //its rotation
                                       positionSand3,					   //its position
                                       fSand3Logical,                                    //its logical volume
                                       "SandDet3",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                       fCheckOverlaps);                                    //checking overlaps


  G4Region* aRegion4 = new G4Region("SandDet3");
  fSand3Logical->SetRegion(aRegion4);
  aRegion4->AddRootLogicalVolume(fSand3Logical);
  
     // ========================================
    //               Sand_paper 3
    // ========================================
    

 
                                 
 fSphere6Solid = new G4Sphere("sphere6",
			      0*mm,                         //its R_min
			      fPixelSize2X/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere6Logical = new G4LogicalVolume(fSphere6Solid, 
				      fSphereMaterial,
				      "SphereLV6");                                     
 
 
 
 
 // Build the Speckel's Envelope 
  fObject5Solid =  new G4Box("SandDet4",                    //its name                 
                             fPixiRadSize2Y/2,              //its size
                             fPixiRadSize2Y/2,
                             fPixiRadSize2Z/2);        
      
  fObject5Logical =  new G4LogicalVolume(fObject5Solid,    //its solid
                                      fWorldMaterial,   //its material
                                      "SandDet4");       //its name
 
  
  
 std::vector<G4ThreeVector> coordinates3 = ReadCoordinates("../data/58_um_Coord3.txt");
 G4int Numb_speck3 = 300000;
 
 for(G4int i = 0; i < Numb_speck3; i ++)
 {
 	
 	  //G4double x1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2; 
 	  //G4double y1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2;
          //G4double z1 = G4UniformRand()*fPixiRadSize2Z - fPixiRadSize2Z/2;
      
      G4ThreeVector position3 = coordinates3[i];// G4ThreeVector(x1, y1, 0);
      //G4ThreeVector position1 =  G4ThreeVector(x1, y1, 0);
      //G4String  names = "Speckle_" + G4UIcommand::ConvertToString(copy_nn);
      
      		fSphere6Physical = new G4PVPlacement(0, 
      						     position3 ,//+ G4ThreeVector(0*um, 0*um, fSourcePosZ+fSrcObjDistance -fTras -0.2*cm),
      					             fSphere6Logical,
      					             "spheresplacement", 
      					             fObject5Logical,
      					             false, 
      					             i, 
      						     fCheckOverlaps); 
       
	//copy_nn++;   
 	 
 }                                             
 
 // - Place the Sandpaper Envelope in the World
  G4ThreeVector positionSand4 = G4ThreeVector(0*um+fTrasX, 0*um+fTrasY, fSourcePosZ+fSrcObjDistance -fTras + 2*fPixiRadSize2Z);
  fObject5Physical = new G4PVPlacement(0,                                                  //its rotation
                                       positionSand4,					   //its position
                                       fObject5Logical,                                    //its logical volume
                                       "SandDet4",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                       fCheckOverlaps);                                    //checking overlaps


  G4Region* aRegion5 = new G4Region("SandDet4");
  fObject5Logical->SetRegion(aRegion4);
  aRegion5->AddRootLogicalVolume(fObject5Logical);

   // ========================================
    //               Sand_paper 4
    // ========================================
    

 
                                 
 fSphere7Solid = new G4Sphere("sphere7",
			      0*mm,                         //its R_min
			      208*um/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere7Logical = new G4LogicalVolume(fSphere7Solid, 
				      fSphereMaterial,
				      "SphereLV7");                                     
 
 
 
 
 // Build the Speckel's Envelope 
  fObject6Solid =  new G4Box("SandDet5",                    //its name                 
                             208*256*um/2,              //its size
                             208*256*um/2,
                             208*um/2);        
      
  fObject6Logical =  new G4LogicalVolume(fObject6Solid,    //its solid
                                      fWorldMaterial,   //its material
                                      "SandDet5");       //its name
 
  
  
  std::vector<G4ThreeVector> coordinates4 = ReadCoordinates("../data/58_um_Coord4.txt");
 G4int Numb_speck4 = 300000;
 
 for(G4int i = 0; i < Numb_speck4; i ++)
 {
 	
 	  //G4double x1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2; 
 	  //G4double y1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2;
          //G4double z1 = G4UniformRand()*fPixiRadSize2Z - fPixiRadSize2Z/2;
      
      G4ThreeVector position4 = coordinates4[i];// G4ThreeVector(x1, y1, 0);
      //G4ThreeVector position1 =  G4ThreeVector(x1, y1, 0);
      //G4String  names = "Speckle_" + G4UIcommand::ConvertToString(copy_nn);
      
      		fSphere7Physical = new G4PVPlacement(0, 
      						     position4 ,//+ G4ThreeVector(0*um, 0*um, fSourcePosZ+fSrcObjDistance -fTras -0.2*cm),
      					             fSphere7Logical,
      					             "spheresplacement", 
      					             fObject6Logical,
      					             false, 
      					             i, 
      						     fCheckOverlaps); 
       
	//copy_nn++;   
 	 
 }                                            
 
 // - Place the Sandpaper Envelope in the World
  G4ThreeVector positionSand5 = G4ThreeVector(0*um+fTrasX, 0*um+fTrasY, fSourcePosZ+fSrcObjDistance -fTras + 3*fPixiRadSize2Z);
  fObject6Physical = new G4PVPlacement(0,                                                  //its rotation
                                       positionSand5,					   //its position
                                       fObject6Logical,                                    //its logical volume
                                       "SandDet5",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                       fCheckOverlaps);                                    //checking overlaps


  G4Region* aRegion6 = new G4Region("SandDet5");
  fObject6Logical->SetRegion(aRegion6);
  aRegion6->AddRootLogicalVolume(fObject6Logical);
  
        // ========================================
    //               Sand_paper 5
    // =========================================

 /*
                                 
 fSphere9Solid = new G4Sphere("sphere9",
			      0*mm,                         //its R_min
			      fPixelSize2X/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere9Logical = new G4LogicalVolume(fSphere9Solid, 
				      fSphereMaterial,
				      "SphereLV9");                                     
 
 
 
 
 // Build the Speckel's Envelope 
  fSand4Solid =  new G4Box("SandDet6",                    //its name                 
                             fPixiRadSize2X/2,              //its size
                             fPixiRadSize2X/2,
                             fPixiRadSize2Z/2);        
      
  fSand4Logical =  new G4LogicalVolume(fSand4Solid,    //its solid
                                      fWorldMaterial,   //its material
                                      "SandDet6");       //its name
 
  
 // G4double x1 = + j*fPixelSize2X - fPixiRadSize2X/2;// + fPixelSizeX/2;
 //      G4double y1 = - i*fPixelSize2X + fPixiRadSize2X/2;// - fPixelSizeY/2;
 
  std::vector<G4ThreeVector> coordinates5 = ReadCoordinates("../data/58_um_Coord5.txt");
 G4int Numb_speck5 = 300000;
 
 for(G4int i = 0; i < Numb_speck5; i ++)
 {
 	
 	  //G4double x1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2; 
 	  //G4double y1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2;
          //G4double z1 = G4UniformRand()*fPixiRadSize2Z - fPixiRadSize2Z/2;
      
      G4ThreeVector position5 = coordinates5[i];// G4ThreeVector(x1, y1, 0);
      //G4ThreeVector position1 =  G4ThreeVector(x1, y1, 0);
      //G4String  names = "Speckle_" + G4UIcommand::ConvertToString(copy_nn);
      
      		fSphere9Physical = new G4PVPlacement(0, 
      						     position5 ,//+ G4ThreeVector(0*um, 0*um, fSourcePosZ+fSrcObjDistance -fTras -0.2*cm),
      					             fSphere9Logical,
      					             "spheresplacement", 
      					             fSand4Logical,
      					             false, 
      					             i, 
      						     fCheckOverlaps); 
       
	//copy_nn++;   
 	 
 }                             
 
 // - Place the Sandpaper Envelope in the World
  G4ThreeVector positionSand6 = G4ThreeVector(0*um+fTrasX, 0*um+fTrasY, fSourcePosZ+fSrcObjDistance -fTras + 4*fPixiRadSize2Z);
  fSand4Physical = new G4PVPlacement(0,                                                  //its rotation
                                       positionSand6,					   //its position
                                       fSand4Logical,                                    //its logical volume
                                       "SandDet6",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                       fCheckOverlaps);                                    //checking overlaps


  G4Region* aRegion7 = new G4Region("SandDet6");
  fSand4Logical->SetRegion(aRegion7);
  aRegion7->AddRootLogicalVolume(fSand4Logical);
  
         // ========================================
    //               Sand_paper 6
    // =========================================

 
                                 
 fSphere10Solid = new G4Sphere("sphere10",
			      0*mm,                         //its R_min
			      fPixelSize2X/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere10Logical = new G4LogicalVolume(fSphere10Solid, 
				      fSphereMaterial,
				      "SphereLV10");                                     
 
 
 
 
 // Build the Speckel's Envelope 
  fSand5Solid =  new G4Box("SandDet7",                    //its name                 
                             fPixiRadSize2X/2,              //its size
                             fPixiRadSize2X/2,
                             fPixiRadSize2Z/2);        
      
  fSand5Logical =  new G4LogicalVolume(fSand5Solid,    //its solid
                                      fWorldMaterial,   //its material
                                      "SandDet7");       //its name
 
  
 // G4double x1 = + j*fPixelSize2X - fPixiRadSize2X/2;// + fPixelSizeX/2;
 //      G4double y1 = - i*fPixelSize2X + fPixiRadSize2X/2;// - fPixelSizeY/2;
 
  std::vector<G4ThreeVector> coordinates6 = ReadCoordinates("../data/58_um_Coord6.txt");
 G4int Numb_speck6 = 300000;
 
 for(G4int i = 0; i < Numb_speck6; i ++)
 {
 	
 	  //G4double x1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2; 
 	  //G4double y1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2;
          //G4double z1 = G4UniformRand()*fPixiRadSize2Z - fPixiRadSize2Z/2;
      
      G4ThreeVector position6 = coordinates6[i];// G4ThreeVector(x1, y1, 0);
      //G4ThreeVector position1 =  G4ThreeVector(x1, y1, 0);
      //G4String  names = "Speckle_" + G4UIcommand::ConvertToString(copy_nn);
      
      		fSphere10Physical = new G4PVPlacement(0, 
      						     position6 ,//+ G4ThreeVector(0*um, 0*um, fSourcePosZ+fSrcObjDistance -fTras -0.2*cm),
      					             fSphere10Logical,
      					             "spheresplacement", 
      					             fSand5Logical,
      					             false, 
      					             i, 
      						     fCheckOverlaps); 
       
	//copy_nn++;   
 	 
 }                             
 
 // - Place the Sandpaper Envelope in the World
  G4ThreeVector positionSand7 = G4ThreeVector(0*um+fTrasX, 0*um+fTrasY, fSourcePosZ+fSrcObjDistance -fTras + 5*fPixiRadSize2Z);
  fSand5Physical = new G4PVPlacement(0,                                                  //its rotation
                                       positionSand7,					   //its position
                                       fSand5Logical,                                    //its logical volume
                                       "SandDet7",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                       fCheckOverlaps);                                    //checking overlaps


  G4Region* aRegion8 = new G4Region("SandDet7");
  fSand5Logical->SetRegion(aRegion8);
  aRegion8->AddRootLogicalVolume(fSand5Logical);

    // ========================================
    //               Sand_paper 7
    // =========================================

 
                                 
 fSphere11Solid = new G4Sphere("sphere11",
			      0*mm,                         //its R_min
			      fPixelSize2X/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere11Logical = new G4LogicalVolume(fSphere11Solid, 
				      fSphereMaterial,
				      "Sphere11LV");                                     
 
 
 
 
 // Build the Speckel's Envelope 
  fSand6Solid =  new G4Box("SandDet8",                    //its name                 
                             fPixiRadSize2X/2,              //its size
                             fPixiRadSize2X/2,
                             fPixiRadSize2Z/2);        
      
  fSand6Logical =  new G4LogicalVolume(fSand6Solid,    //its solid
                                      fWorldMaterial,   //its material
                                      "SandDet8");       //its name
 
  
 // G4double x1 = + j*fPixelSize2X - fPixiRadSize2X/2;// + fPixelSizeX/2;
 //      G4double y1 = - i*fPixelSize2X + fPixiRadSize2X/2;// - fPixelSizeY/2;
 
  std::vector<G4ThreeVector> coordinates7 = ReadCoordinates("../data/58_um_Coord7.txt");
 G4int Numb_speck7 = 300000;
 
 for(G4int i = 0; i < Numb_speck7; i ++)
 {
 	
 	  //G4double x1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2; 
 	  //G4double y1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2;
          //G4double z1 = G4UniformRand()*fPixiRadSize2Z - fPixiRadSize2Z/2;
      
      G4ThreeVector position7 = coordinates7[i];// G4ThreeVector(x1, y1, 0);
      //G4ThreeVector position1 =  G4ThreeVector(x1, y1, 0);
      //G4String  names = "Speckle_" + G4UIcommand::ConvertToString(copy_nn);
      
      		fSphere11Physical = new G4PVPlacement(0, 
      						          position7 ,//+ G4ThreeVector(0*um, 0*um, fSourcePosZ+fSrcObjDistance -fTras -0.2*cm),
      					             fSphere11Logical,
      					             "spheresplacement7", 
      					             fSand6Logical,
      					             false, 
      					             i, 
      						     fCheckOverlaps); 
       
	//copy_nn++;   
 	 
 }                             
 
 // - Place the Sandpaper Envelope in the World
  G4ThreeVector positionSand8 = G4ThreeVector(0*um+fTrasX, 0*um+fTrasY, fSourcePosZ+fSrcObjDistance -fTras + 6*fPixiRadSize2Z);
  fSand6Physical = new G4PVPlacement(0,                                                  //its rotation
                                       positionSand8,					   //its position
                                       fSand6Logical,                                    //its logical volume
                                       "SandDet8",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                       fCheckOverlaps);                                    //checking overlaps


  G4Region* aRegion9 = new G4Region("SandDet8");
  fSand6Logical->SetRegion(aRegion9);
  aRegion9->AddRootLogicalVolume(fSand6Logical);

    // ========================================
    //               Sand_paper 8
    // =========================================

 
                                 
 fSphere12Solid = new G4Sphere("sphere12",
			      0*mm,                         //its R_min
			      fPixelSize2X/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere12Logical = new G4LogicalVolume(fSphere12Solid, 
				      fSphereMaterial,
				      "Sphere12LV");                                     
 
 
 
 
 // Build the Speckel's Envelope 
  fSand7Solid =  new G4Box("SandDet9",                    //its name                 
                             fPixiRadSize2X/2,              //its size
                             fPixiRadSize2X/2,
                             fPixiRadSize2Z/2);        
      
  fSand7Logical =  new G4LogicalVolume(fSand7Solid,    //its solid
                                      fWorldMaterial,   //its material
                                      "SandDet9");       //its name
 
  
 // G4double x1 = + j*fPixelSize2X - fPixiRadSize2X/2;// + fPixelSizeX/2;
 //      G4double y1 = - i*fPixelSize2X + fPixiRadSize2X/2;// - fPixelSizeY/2;
 
  std::vector<G4ThreeVector> coordinates8 = ReadCoordinates("../data/58_um_Coord8.txt");
 G4int Numb_speck8 = 300000;
 
 for(G4int i = 0; i < Numb_speck8; i ++)
 {
 	
 	  //G4double x1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2; 
 	  //G4double y1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2;
          //G4double z1 = G4UniformRand()*fPixiRadSize2Z - fPixiRadSize2Z/2;
      
      G4ThreeVector position8 = coordinates8[i];// G4ThreeVector(x1, y1, 0);
      //G4ThreeVector position1 =  G4ThreeVector(x1, y1, 0);
      //G4String  names = "Speckle_" + G4UIcommand::ConvertToString(copy_nn);
      
      		fSphere12Physical = new G4PVPlacement(0, 
      						     position8 ,//+ G4ThreeVector(0*um, 0*um, fSourcePosZ+fSrcObjDistance -fTras -0.2*cm),
      					             fSphere12Logical,
      					             "spheresplacement8", 
      					             fSand7Logical,
      					             false, 
      					             i, 
      						     fCheckOverlaps); 
       
	//copy_nn++;   
 	 
 }                             
 
 // - Place the Sandpaper Envelope in the World
  G4ThreeVector positionSand9 = G4ThreeVector(0*um+fTrasX, 0*um+fTrasY, fSourcePosZ+fSrcObjDistance -fTras + 7*fPixiRadSize2Z);
  fSand7Physical = new G4PVPlacement(0,                                                  //its rotation
                                       positionSand9,					   //its position
                                       fSand7Logical,                                    //its logical volume
                                       "SandDet9",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                       fCheckOverlaps);                                    //checking overlaps


  G4Region* aRegion10 = new G4Region("SandDet9");
  fSand7Logical->SetRegion(aRegion10);
  aRegion10->AddRootLogicalVolume(fSand7Logical);

  //  ======================================
    //               Sand_paper 9
    // =========================================

 
                                 
 fSphere13Solid = new G4Sphere("sphere13",
			      0*mm,                         //its R_min
			      fPixelSize2X/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere13Logical = new G4LogicalVolume(fSphere13Solid, 
				      fSphereMaterial,
				      "Sphere13LV");                                     
 
 
 
 
 // Build the Speckel's Envelope 
  fSand8Solid =  new G4Box("SandDet10",                    //its name                 
                             fPixiRadSize2X/2,              //its size
                             fPixiRadSize2X/2,
                             fPixiRadSize2Z/2);        
      
  fSand8Logical =  new G4LogicalVolume(fSand8Solid,    //its solid
                                      fWorldMaterial,   //its material
                                      "SandDet10");       //its name
 
  
 // G4double x1 = + j*fPixelSize2X - fPixiRadSize2X/2;// + fPixelSizeX/2;
 //      G4double y1 = - i*fPixelSize2X + fPixiRadSize2X/2;// - fPixelSizeY/2;
 
  std::vector<G4ThreeVector> coordinates9 = ReadCoordinates("../data/58_um_Coord9.txt");
 G4int Numb_speck9 = 300000;
 
 for(G4int i = 0; i < Numb_speck9; i ++)
 {
 	
 	  //G4double x1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2; 
 	  //G4double y1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2;
          //G4double z1 = G4UniformRand()*fPixiRadSize2Z - fPixiRadSize2Z/2;
      
      G4ThreeVector position9 = coordinates9[i];// G4ThreeVector(x1, y1, 0);
      //G4ThreeVector position1 =  G4ThreeVector(x1, y1, 0);
      //G4String  names = "Speckle_" + G4UIcommand::ConvertToString(copy_nn);
      
      		fSphere13Physical = new G4PVPlacement(0, 
      						     position9 ,//+ G4ThreeVector(0*um, 0*um, fSourcePosZ+fSrcObjDistance -fTras -0.2*cm),
      					             fSphere13Logical,
      					             "spheresplacement9", 
      					             fSand8Logical,
      					             false, 
      					             i, 
      						     fCheckOverlaps); 
       
	//copy_nn++;   
 	 
 }                             
 
 // - Place the Sandpaper Envelope in the World
  G4ThreeVector positionSand10 = G4ThreeVector(0*um+fTrasX, 0*um+fTrasY, fSourcePosZ+fSrcObjDistance -fTras + 8*fPixiRadSize2Z);
  fSand8Physical = new G4PVPlacement(0,                                                  //its rotation
                                       positionSand10,					   //its position
                                       fSand8Logical,                                    //its logical volume
                                       "SandDet10",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                       fCheckOverlaps);                                    //checking overlaps


  G4Region* aRegion11 = new G4Region("SandDet10");
  fSand8Logical->SetRegion(aRegion11);
  aRegion11->AddRootLogicalVolume(fSand8Logical);

  //  ======================================
    //               Sand_paper 10
    // =========================================

 
                                 
 fSphere14Solid = new G4Sphere("sphere14",
			      0*mm,                         //its R_min
			      fPixelSize2X/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere14Logical = new G4LogicalVolume(fSphere14Solid, 
				      fSphereMaterial,
				      "Sphere14LV");                                     
 
 
 
 
 // Build the Speckel's Envelope 
  fSand9Solid =  new G4Box("SandDet11",                    //its name                 
                             fPixiRadSize2X/2,              //its size
                             fPixiRadSize2X/2,
                             fPixiRadSize2Z/2);        
      
  fSand9Logical =  new G4LogicalVolume(fSand9Solid,    //its solid
                                      fWorldMaterial,   //its material
                                      "SandDet11");       //its name
 
  
 // G4double x1 = + j*fPixelSize2X - fPixiRadSize2X/2;// + fPixelSizeX/2;
 //      G4double y1 = - i*fPixelSize2X + fPixiRadSize2X/2;// - fPixelSizeY/2;
 
  std::vector<G4ThreeVector> coordinates10 = ReadCoordinates("../data/58_um_Coord10.txt");
 G4int Numb_speck10 = 300000;
 
 for(G4int i = 0; i < Numb_speck10; i ++)
 {
 	
 	  //G4double x1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2; 
 	  //G4double y1 = G4UniformRand()*fPixiRadSize2X - fPixiRadSize2X/2;
          //G4double z1 = G4UniformRand()*fPixiRadSize2Z - fPixiRadSize2Z/2;
      
      G4ThreeVector position10 = coordinates10[i];// G4ThreeVector(x1, y1, 0);
      //G4ThreeVector position1 =  G4ThreeVector(x1, y1, 0);
      //G4String  names = "Speckle_" + G4UIcommand::ConvertToString(copy_nn);
      
      		fSphere14Physical = new G4PVPlacement(0, 
      						     position10,//+ G4ThreeVector(0*um, 0*um, fSourcePosZ+fSrcObjDistance -fTras -0.2*cm),
      					             fSphere14Logical,
      					             "spheresplacement10", 
      					             fSand9Logical,
      					             false, 
      					             i, 
      						     fCheckOverlaps); 
       
	//copy_nn++;   
 	 
 }                             
 
 // - Place the Sandpaper Envelope in the World
  G4ThreeVector positionSand11 = G4ThreeVector(0*um+fTrasX, 0*um+fTrasY, fSourcePosZ+fSrcObjDistance -fTras + 9*fPixiRadSize2Z);
  fSand9Physical = new G4PVPlacement(0,                                                  //its rotation
                                       positionSand11,					   //its position
                                       fSand9Logical,                                    //its logical volume
                                       "SandDet11",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                       fCheckOverlaps);                                    //checking overlaps


  G4Region* aRegion12 = new G4Region("SandDet11");
  fSand9Logical->SetRegion(aRegion12);
  aRegion12->AddRootLogicalVolume(fSand9Logical);

*/
     // ========================================
    //              Envelope Cellulose Layer
    // ========================================
//-------------------------------------------------------------
 
 // Build the Speckel's Envelope 
  fSand2Solid =  new G4Box("SandDet2",                    //its name                 
                             fPixiRadSize2Y/2,              //its size
                             fPixiRadSize2Y/2,
                             fPixiRadSize4Z/2);        
      
  fSand2Logical =  new G4LogicalVolume(fSand2Solid,    //its solid
                                      fMuscleMaterial,   //its material
                                      "SandDet2");       //its name
 
// -----------------------------------------------------------------  

 // - Place the Sandpaper Envelope in the World
  G4ThreeVector positionSand2 = G4ThreeVector(0*um+fTrasX, 0*um+fTrasY, fSourcePosZ+fSrcObjDistance -fTras + 19*fPixiRadSize2Z/2 + fPixiRadSize4Z/2);
  fSand2Physical = new G4PVPlacement(0,                                                  //its rotation
                                       positionSand2,					   //its position
                                       fSand2Logical,                                    //its logical volume
                                       "SandDet2",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                       fCheckOverlaps);                                    //checking overlaps

 
 
/*
  // ========================================
  //                DETECTOR MASK M2 and SUBSTRATE
  // ========================================

  if (fAcquisitionType=="doublemask")
  {
  G4double mag_M2 = (fSrcObjDistance+fObjectDetDistance)/(fSrcObjDistance+fObjectDetDistance-(fMaskThickness/2+fPixiRadSizeZ/2)); // magnification of the mask M2
  G4ThreeVector M2Position = positionPixirad-G4ThreeVector(0*cm,0,(fMaskThickness+fPixiRadSizeZ)/2+1*nm);
  // - Build the MASK APERTURE UNIT as a Box
  CreateMask("M2", mag_M2,fM2Pitch, fM2Aperture, M2Position, fMaskThickness, fMaskMaterial, fEnvelopeM2Logical, fEnvelopeM2Physical);
  CreateSubstrate("M2sub", mag_M2, M2Position-G4ThreeVector(0,0,fSubThickness/2+fMaskThickness/2), fSubThickness, fSubMaterial, fM2subLogical, fM2subPhysical);
  }
 */ 

  
  // ========================================
  //                 Objects
  // ========================================




  // -------Build the PMMA phantom box ----------------------------------------

  fObjectSolid =  new G4Box("PMMA_BOX",                    //its name                 
                             fPixiRadSize3X/2,              //its size
                             fPixiRadSize3Y/2,
                             fPixiRadSize3Z/2);        
      
  fObjectLogical =  new G4LogicalVolume(fObjectSolid,    //its solid
                                         fObjectMaterial,   //its material
                                         "PMMA_BOX");       //its name
                                         
                                         
                                         
                                         

                           
  G4ThreeVector positionPMMA = G4ThreeVector(0, 0, fSourcePosZ+fSrcObjDistance);
  fObjectPhysical = new G4PVPlacement(0,                                                  //its rotation
                                       positionPMMA,					   //its position
                                       fObjectLogical,                                    //its logical volume
                                       "PMMA_BOX",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                       fCheckOverlaps);                                    //checking overlaps

// -------------------------------------------------------------------------------


// ----BUILD A WAX BOX ----------------------------------------------------------  
  
fObject2Solid =  new G4Box("WaxBox",                    //its name                 
                             fPixiRadSize3X/2,              //its size
                             fPixiRadSize3Y/2,
                            0.214*fPixiRadSize3Z/2);        
      
  fObject2Logical =  new G4LogicalVolume(fObject2Solid,    //its solid
                                         fObject2Material,   //its material
                                         "WaxBox");       //its name
                                         
                                         
                                         
                                         

                           
  G4ThreeVector positionWax = G4ThreeVector(0, 0, fSourcePosZ+fSrcObjDistance- fPixiRadSize3Z/2 -0.214*fPixiRadSize3Z/2);
  fObject2Physical = new G4PVPlacement(0,                                                  //its rotation
                                       positionWax,					   //its position
                                       fObject2Logical,                                    //its logical volume
                                       "WaxBox",                                          //its name
                                       fWorldLogical,                                      //its mother volume
                                       false,                                              //no boolean operation
                                       0,                                                  //copy number
                                      fCheckOverlaps);                                    //checking overlaps

                                      
// -----------------NYLON FIBER----------------------------------------------------------------
/*
 fObject3Solid = new G4Box("Trap3",                         //its name
                            0.2*mm,                             //its half x1
                            1*cm,                     //its half x2
                            0.2*mm                     //its half y1
                        //    0.5*fObjSizeY,                     //its half y2
                        //    fObjSizeR
                        );                //its half height
                            
  fObject3Logical = new G4LogicalVolume(fObject3Solid,       //its solid
  					fObject3Material,       //fObject3Material,     //its material
                                       "Trap3LV");          //its name
  
  G4ThreeVector objectPosition3 = G4ThreeVector(0, 0, 0 );
  
  
  G4RotationMatrix* rotMat2 =  new G4RotationMatrix();
  rotMat2->rotateZ(45*deg);
  //rotMat2->rotateY(45*deg);
  
  
  fObject3Physical = new G4PVPlacement(rotMat2,              //its rotation
                                      objectPosition3,       //its translation
                                      fObject3Logical,      //its logical volume
                                      "Trap3",              //its name
                                      fObject2Logical,       //its mother volume
                                      false,               //no boolean operation
                                      0,                   //copy number
                                      fCheckOverlaps);     //checking overlaps



*/
//-------  BUILD A 3mm PMMA thick cover ----------------------------------------------------                                    
                                       

fSphere2Solid = new G4Box("Cover",
 			      fPixiRadSize3X/2,              //its size
                             fPixiRadSize3Y/2,
                             0.0888*fPixiRadSize3Z/2); 
                             
fSphere2Logical = new G4LogicalVolume(fSphere2Solid,
				       fObjectMaterial,
				       "CoverLv");
				       
				       
				       	    		      				                                     
 G4ThreeVector objectPosition4 = G4ThreeVector(0, 0, fSourcePosZ+fSrcObjDistance-fPixiRadSize3Z/2 -0.214*fPixiRadSize3Z - 0.0888*fPixiRadSize3Z/2); 
 // G4RotationMatrix* rotMat2 =  new G4RotationMatrix();
 // rotMat2->rotateZ(fRotAngle);
  
 
 
 fSphere2Physical = new G4PVPlacement(0,              //its rotation
                                      objectPosition4,       //its translation
                                      fSphere2Logical,      //its logical volume
                                      "tube1",              //its name
                                      fWorldLogical,       //its mother volume
                                      false,               //no boolean operation
                                      0,                   //copy number
                                      fCheckOverlaps);     //checking overlaps  
                                 
// ------------------------ end objects ----------------------------------------  
/*
//==================================
 //   FILTERS
 //==================================

  fObject5Solid = new G4Box("Filter",                               //its name
                          10*cm/2,   //its size
                          10*cm/2,
                          fXd/2);

  fObject5Logical = new G4LogicalVolume(fObject5Solid,       //its solid
                                       fObject5Material,    //its material
                                       "FilterLV");          //its name
  G4ThreeVector objectPosition5 = G4ThreeVector(0,0,3*cm+fSourcePosZ);  
  
    fObject6Solid = new G4Box("Filter2",                               //its name
                          10*cm/2,   //its size
                          10*cm/2,
                          fXd2/2);

  fObject6Logical = new G4LogicalVolume(fObject6Solid,       //its solid
                                       fObject6Material,    //its material
                                       "Filter2LV");          //its name
  G4ThreeVector objectPosition6 = G4ThreeVector(0,0,3*cm+fSourcePosZ+fXd2/2+fXd/2);   
 
  fObject6Physical = new G4PVPlacement(0,              //its rotation
                                      objectPosition6,       //its translation
                                     fObject6Logical,      //its logical volume
                                      "Filter2",              //its name
                                      fWorldLogical,       //its mother volume
                                      false,               //no boolean operation
                                      0,                   //copy number
                                      fCheckOverlaps);     //checking overlaps



  fObject5Physical = new G4PVPlacement(0,              //its rotation
                                      objectPosition5,       //its translation
                                     fObject5Logical,      //its logical volume
                                      "Filter",              //its name
                                      fWorldLogical,       //its mother volume
                                      false,               //no boolean operation
                                      0,                   //copy number
                                      fCheckOverlaps);     //checking overlaps
*/
                                   
 //==================================
 //   A TUBE SAMPLE PMMA
 //==================================
/* 
  G4RotationMatrix* rotMat2 =  new G4RotationMatrix();
  rotMat2->rotateX(90*deg);
  
  fSphere2Solid = new G4Tubs("tube1",
 			      0 *cm,
	    		      0.194/2 *cm,
	    		      5*cm,
	    		      0,
	    		      2*pi);
fSphere2Logical = new G4LogicalVolume(fSphere2Solid,
				       fObjectMaterial,
				       "TubeLV");
				       	    		      				                                     
 G4ThreeVector objectPosition4 = G4ThreeVector(0*mm, 0, fSourcePosZ+fSrcObjDistance); 
 
 fSphere2Physical = new G4PVPlacement(rotMat2,              //its rotation
                                      objectPosition4,       //its translation
                                      fSphere2Logical,      //its logical volume
                                      "tube1",              //its name
                                      fWorldLogical,       //its mother volume
                                      false,               //no boolean operation
                                      0,                   //copy number
                                      fCheckOverlaps);     //checking overlaps    

*/
/*                                    
//==================================
 //   A TUBE SAMPLE Blood
 //==================================
  
  fSphere7Solid = new G4Tubs("tube2",
 			      0.0 *cm,
	    		      0.185/2 *cm,
	    		      5*cm,
	    		      0*deg,
	    		      360*deg);
fSphere7Logical = new G4LogicalVolume(fSphere7Solid,
				       fWorldMaterial,
				       "TubeLV2");
				       	    		      				                                     
 G4ThreeVector objectPosition7 = G4ThreeVector(0*mm, 0, fSourcePosZ+fSrcObjDistance); 
 
 fSphere7Physical = new G4PVPlacement(rotMat2,              //its rotation
                                      objectPosition7,       //its translation
                                      fSphere7Logical,      //its logical volume
                                      "tube2",              //its name
                                      fWorldLogical,       //its mother volume
                                      false,               //no boolean operation
                                      0,                   //copy number
                                      fCheckOverlaps);     //checking overlaps                                            
                                      



//===========================================
//            HPA SAMPLE
//===========================================

fSphere6Solid = new G4Tubs("tube3",
			      0.0*cm,                         //its R_min
			      0.185/2*cm,   			     // its R_max
			      5*cm,		            //its symetric angle azim
			      0*deg,                     // its final angle 
			      45*deg);                       //its symetric zenith angle)//its final angle zenith


fSphere6Logical = new G4LogicalVolume(fSphere6Solid, 
				      fObject6Material,
				      "Tube3LV");
G4ThreeVector objectPositionSph = G4ThreeVector(0*mm, 0, 0);

fSphere6Physical = new G4PVPlacement(0, 
				    objectPositionSph,
				    fSphere6Logical, 
				    "Tube3PV",
				    fSphere7Logical, 
				    false, 
				    0,
				    fCheckOverlaps);


*/
				      
//===========================================
//            A SILICON CIRCLE TO RETRIEVE uCs
//===========================================

//==============================
//    540 um        
//==============================

fSphere8Solid = new G4Sphere("sphere8",
                              0*mm,                         //its R_min
                              0.54*mm/2,                               // its R_max
                              0*deg,                        //its symetric angle azim
                              360*deg,                     // its final angle 
                              0*deg,                       //its symetric zenith angle
                              360*deg);                     //its final angle zenith


fSphere8Logical = new G4LogicalVolume(fSphere8Solid, 
                                      fSubMaterial,
                                      "SphereLV8");
G4ThreeVector objectPositionSph8 = G4ThreeVector(-3.52*mm, 0*mm, 0);

fSphere8Physical = new G4PVPlacement(0, 
                                    objectPositionSph8,
                                    fSphere8Logical, 
                                    "spherePV8",
                                    fObject2Logical, 
                                    false, 
                                    0,
                                    fCheckOverlaps);


//===========================================
//           400 um
//===========================================                                    

fSphere9Solid = new G4Sphere("sphere9",
			      0*mm,                         //its R_min
			      0.4*mm/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere9Logical = new G4LogicalVolume(fSphere9Solid, 
				      fSubMaterial,
				      "SphereLV9");

G4ThreeVector objectPositionSph9 = G4ThreeVector(0*mm, 0*mm, 0);

fSphere9Physical = new G4PVPlacement(0, 
                                    objectPositionSph9,
                                    fSphere9Logical, 
                                    "spherePV9",
                                    fObject2Logical, 
                                    false, 
                                    0,
                                    fCheckOverlaps);      

//===========================================
//           320 um
//===========================================  

fSphere10Solid = new G4Sphere("sphere10",
			      0*mm,                         //its R_min
			      320*um/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere10Logical = new G4LogicalVolume(fSphere10Solid, 
				      fSubMaterial,
				      "SphereLV10");  
G4ThreeVector objectPositionSph10 = G4ThreeVector(3.52*mm, 0*mm, 0);               
fSphere10Physical = new G4PVPlacement(0, 
                                    objectPositionSph10,
                                    fSphere10Logical, 
                                    "spherePV10",
                                    fObject2Logical, 
                                    false, 
                                    0,
                                    fCheckOverlaps); 
//===========================================
//           240 um
//===========================================

fSphere11Solid = new G4Sphere("sphere11",
			      0*mm,                         //its R_min
			      500*um/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere11Logical = new G4LogicalVolume(fSphere11Solid, 
				      fSubMaterial,
				      "Sphere11LV");  
G4ThreeVector objectPositionSph11 = G4ThreeVector(-3.52*mm, 3.693*mm, 0);               
fSphere11Physical = new G4PVPlacement(0, 
                                    objectPositionSph11,
                                    fSphere11Logical, 
                                    "spherePV11",
                                    fObject2Logical, 
                                    false, 
                                    0,
                                    fCheckOverlaps);  

//===========================================
//           160 um
//===========================================

fSphere12Solid = new G4Sphere("sphere12",
			      0*mm,                         //its R_min
			      500*um/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere12Logical = new G4LogicalVolume(fSphere12Solid, 
				      fSubMaterial,
				      "Sphere12LV");  
G4ThreeVector objectPositionSph12 = G4ThreeVector(0*mm, 3.693*mm, 0);               
fSphere12Physical = new G4PVPlacement(0, 
                                    objectPositionSph12,
                                    fSphere12Logical, 
                                    "spherePV12",
                                    fObject2Logical, 
                                    false, 
                                    0,
                                    fCheckOverlaps);
//===========================================
//           100 um
//===========================================

fSphere13Solid = new G4Sphere("sphere13",
			      0*mm,                         //its R_min
			      500*um/2,               // its R_max
			      0*deg,		            //its symetric angle azim
			      360*deg,                     // its final angle 
			      0*deg,                       //its symetric zenith angle
			      360*deg);                     //its final angle zenith


fSphere13Logical = new G4LogicalVolume(fSphere13Solid, 
				      fSubMaterial,
				      "Sphere13LV");  
G4ThreeVector objectPositionSph13 = G4ThreeVector(3.52*mm, 3.693*mm, 0);               
fSphere13Physical = new G4PVPlacement(0, 
                                    objectPositionSph13,
                                    fSphere13Logical, 
                                    "spherePV13",
                                    fObject2Logical, 
                                    false, 
                                    0,
                                    fCheckOverlaps);                                    


/*
  // ========================================
  //       SAMPLE MASK M1 and substrate
  // ========================================
  if (fAcquisitionType=="doublemask"|| fAcquisitionType=="singlemask")
  {
  G4double mag_M1 = (fSrcObjDistance+fObjectDetDistance)/(fSrcObjDistance-(fMaskThickness/2+fObjSizeR)); // magnification of the mask M1
  G4ThreeVector M1Position = objectPosition-G4ThreeVector(0,0,(fMaskThickness+2*fObjSizeR)/2)+G4ThreeVector(0*cm,0,0);
  
  std::tie(fEnvelopeM1Logical,fEnvelopeM1Physical) = CreateMask("M1", mag_M1,fM2Pitch, fM2Aperture, M1Position, fMaskThickness, fMaskMaterial, fEnvelopeM1Logical, fEnvelopeM1Physical);
  
  std::tie(fM1subLogical,fM1subPhysical) = CreateSubstrate("M1sub", mag_M1, M1Position-G4ThreeVector(0,0,fSubThickness/2+fMaskThickness/2), fSubThickness, fSubMaterial, fM1subLogical,fM1subPhysical);
  }
*/
  // ========================================
  //               ION CHAMBER
  // ========================================

  // - Build the ION CHAMBER as an unrotated Box
  fIonCSolid = new G4Box("IonChamber",                     //its name
                         fPixiRadSizeX/2,                            //its size
                         fPixiRadSizeY/2,
                         2*mm);          
   
  fIonCLogical = new G4LogicalVolume(fIonCSolid,          //its solid
                                     fIonCMaterial,       //its material
                                     "IonChamberLV");     //its name
                       
  G4ThreeVector IOCposition = positionPixirad - G4ThreeVector(0, 0, 10*mm);
  fIonCPhysical =  new G4PVPlacement(0,                   //no rotation
                                     IOCposition,            //at position
                                     fIonCLogical,        //its logical volume
                                     "IonChamber",        //its name
                                     fWorldLogical,       //its mother  volume
                                     false,               //no boolean operation
                                     0,                   //copy number
                                     fCheckOverlaps);     //checking overlaps 


  
  // ========================================
  //              VISUALIZATION
  // ========================================

  G4VisAttributes* worldVisAtt = new G4VisAttributes(G4Colour(1.0,1.0,1.0));
  fWorldLogical->SetVisAttributes(worldVisAtt);  
  // logicWorld->SetVisAttributes(G4VisAttributes::GetInvisible());  

  G4VisAttributes* objectVisAtt = new G4VisAttributes(G4Colour(0.0,0.0,1.0,0.5));
//  objectVisAtt->SetForceWireframe(true);
  objectVisAtt->SetForceSolid(true);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fObjectLogical->SetVisAttributes(objectVisAtt);
 
  G4VisAttributes* objectVisAtt2 = new G4VisAttributes(G4Colour(1.0,0.0,0.0,0.5));
//  objectVisAtt2->SetForceWireframe(true);
  objectVisAtt2->SetForceSolid(true);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fObject2Logical->SetVisAttributes(objectVisAtt2);


  G4VisAttributes* sphere2VisAtt = new G4VisAttributes(G4Colour(0.6,0.8,1.0));
//  objectVisAtt2->SetForceWireframe(true);
  sphere2VisAtt->SetForceSolid(true);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere2Logical->SetVisAttributes(sphere2VisAtt);

 G4VisAttributes* sphere8VisAtt = new G4VisAttributes(G4Colour(0.6,0.8,1.0));
//  objectVisAtt2->SetForceWireframe(true);
  sphere8VisAtt->SetForceSolid(true);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere8Logical->SetVisAttributes(sphere8VisAtt);
  
  G4VisAttributes* ionCVisAtt = new G4VisAttributes(G4Colour(1.0,0.0,0.0));
  ionCVisAtt->SetVisibility(false);
  ionCVisAtt->SetForceWireframe(true);
  fIonCLogical->SetVisAttributes(ionCVisAtt);  
 
  G4VisAttributes* pixelVisAtt = new G4VisAttributes(G4Colour(0.0,0,1.0));
  pixelVisAtt->SetForceWireframe(true);
  //pixelVisAtt->SetForceSolid(true);
  pixelVisAtt->SetVisibility(false);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  fPixelLogical->SetVisAttributes(pixelVisAtt);

  G4VisAttributes* pixiradVisAtt = new G4VisAttributes(G4Colour(0.0,0,1.0));
  pixiradVisAtt->SetForceWireframe(true);
  pixiradVisAtt->SetVisibility(true);
  pixiradVisAtt->SetForceSolid(true);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  fPixiRadLogical->SetVisAttributes(pixiradVisAtt);

G4VisAttributes* sphere9VisAtt = new G4VisAttributes(G4Colour(0.6,0.8,1.0));
//  objectVisAtt2->SetForceWireframe(true);
  sphere9VisAtt->SetForceSolid(true);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere9Logical->SetVisAttributes(sphere9VisAtt);

G4VisAttributes* sphere10VisAtt = new G4VisAttributes(G4Colour(0.6,0.8,1.0));
//  objectVisAtt2->SetForceWireframe(true);
  sphere10VisAtt->SetForceSolid(true);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere10Logical->SetVisAttributes(sphere10VisAtt);  

  G4VisAttributes* sphere11VisAtt = new G4VisAttributes(G4Colour(0.6,0.8,1.0));
//  objectVisAtt2->SetForceWireframe(true);
  sphere11VisAtt->SetForceSolid(true);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere11Logical->SetVisAttributes(sphere11VisAtt);  

  G4VisAttributes* sphere12VisAtt = new G4VisAttributes(G4Colour(0.6,0.8,1.0));
//  objectVisAtt2->SetForceWireframe(true);
  sphere12VisAtt->SetForceSolid(true);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere12Logical->SetVisAttributes(sphere12VisAtt); 

   G4VisAttributes* sphere13VisAtt = new G4VisAttributes(G4Colour(0.6,0.8,1.0));
//  objectVisAtt2->SetForceWireframe(true);
  sphere13VisAtt->SetForceSolid(true);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere13Logical->SetVisAttributes(sphere13VisAtt);    
 //================= SAND PAPER VIS =========================================
 
  G4VisAttributes* sandVisAtt = new G4VisAttributes(G4Colour(1.0,0.0,0.0));
  sandVisAtt->SetForceWireframe(true);
  sandVisAtt->SetVisibility(true);
  sandVisAtt->SetForceSolid(true);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  fSandLogical->SetVisAttributes(sandVisAtt);
  
  G4VisAttributes* sphereVisAtt = new G4VisAttributes(G4Colour(1.0,0.0,0.0));
  sphereVisAtt->SetForceWireframe(true);
  //sphereVisAtt->SetForceSolid(true);
  sphereVisAtt->SetVisibility(false);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphereLogical->SetVisAttributes(sphereVisAtt);

  G4VisAttributes* sand2VisAtt = new G4VisAttributes(G4Colour(0.0,0,1.0));
  sand2VisAtt->SetForceWireframe(true);
  sand2VisAtt->SetVisibility(true);
  sand2VisAtt->SetForceSolid(true);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  fSand2Logical->SetVisAttributes(sand2VisAtt);
 
  G4VisAttributes* sand3VisAtt = new G4VisAttributes(G4Colour(0.0,1.0,0.0));
  sand3VisAtt->SetForceWireframe(true);
  sand3VisAtt->SetVisibility(true);
  sand3VisAtt->SetForceSolid(true);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  fSand3Logical->SetVisAttributes(sand3VisAtt);
  
  G4VisAttributes* sphere5VisAtt = new G4VisAttributes(G4Colour(0.0,1.0,0.0));
  sphere5VisAtt->SetForceWireframe(true);
  //sphereVisAtt->SetForceSolid(true);
  sphere5VisAtt->SetVisibility(false);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere5Logical->SetVisAttributes(sphere5VisAtt);

  G4VisAttributes* sand4VisAtt = new G4VisAttributes(G4Colour(1.0,1.0,0.0));
  sand4VisAtt->SetForceWireframe(true);
  sand4VisAtt->SetVisibility(true);
  sand4VisAtt->SetForceSolid(true);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  fObject5Logical->SetVisAttributes(sand4VisAtt);
  
  G4VisAttributes* sphere6VisAtt = new G4VisAttributes(G4Colour(0.0,1.0,0.0));
  sphere6VisAtt->SetForceWireframe(true);
  //sphereVisAtt->SetForceSolid(true);
  sphere6VisAtt->SetVisibility(false);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere6Logical->SetVisAttributes(sphere6VisAtt);
  
  G4VisAttributes* sand5VisAtt = new G4VisAttributes(G4Colour(0.0,1.0,1.0));
  sand5VisAtt->SetForceWireframe(true);
  sand5VisAtt->SetVisibility(true);
  sand5VisAtt->SetForceSolid(true);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  fObject6Logical->SetVisAttributes(sand5VisAtt);
  
  G4VisAttributes* sphere7VisAtt = new G4VisAttributes(G4Colour(0.0,1.0,0.0));
  sphere7VisAtt->SetForceWireframe(true);
  //sphereVisAtt->SetForceSolid(true);
  sphere7VisAtt->SetVisibility(false);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere7Logical->SetVisAttributes(sphere7VisAtt);

  /*
  G4VisAttributes* sand6VisAtt = new G4VisAttributes(G4Colour(1.0,1.0,1.0));
  sand6VisAtt->SetForceWireframe(true);
  sand6VisAtt->SetVisibility(true);
  sand6VisAtt->SetForceSolid(true);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  fSand4Logical->SetVisAttributes(sand6VisAtt);
  
  G4VisAttributes* sphere9VisAtt = new G4VisAttributes(G4Colour(1.0,1.0,1.0));
  sphere9VisAtt->SetForceWireframe(true);
  //sphereVisAtt->SetForceSolid(true);
  sphere9VisAtt->SetVisibility(false);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere9Logical->SetVisAttributes(sphere9VisAtt);
  
  G4VisAttributes* sand7VisAtt = new G4VisAttributes(G4Colour(1.0,1.0,1.0));
  sand7VisAtt->SetForceWireframe(true);
  sand7VisAtt->SetVisibility(true);
  sand7VisAtt->SetForceSolid(true);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  fSand5Logical->SetVisAttributes(sand7VisAtt);
  
  G4VisAttributes* sphere10VisAtt = new G4VisAttributes(G4Colour(1.0,1.0,1.0));
  sphere10VisAtt->SetForceWireframe(true);
  //sphereVisAtt->SetForceSolid(true);
  sphere10VisAtt->SetVisibility(false);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere10Logical->SetVisAttributes(sphere10VisAtt);

  G4VisAttributes* sand8VisAtt = new G4VisAttributes(G4Colour(1.0,1.0,1.0));
  sand8VisAtt->SetForceWireframe(true);
  sand8VisAtt->SetVisibility(true);
  sand8VisAtt->SetForceSolid(true);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  fSand6Logical->SetVisAttributes(sand8VisAtt);

  
  G4VisAttributes* sphere11VisAtt = new G4VisAttributes(G4Colour(1.0,1.0,1.0));
  sphere11VisAtt->SetForceWireframe(true);
  //sphereVisAtt->SetForceSolid(true);
  sphere11VisAtt->SetVisibility(false);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere11Logical->SetVisAttributes(sphere11VisAtt);

  G4VisAttributes* sand9VisAtt = new G4VisAttributes(G4Colour(1.0,0,0));
  sand9VisAtt->SetForceWireframe(true);
  sand9VisAtt->SetVisibility(true);
  sand9VisAtt->SetForceSolid(true);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  fSand7Logical->SetVisAttributes(sand9VisAtt);
  
  G4VisAttributes* sphere12VisAtt = new G4VisAttributes(G4Colour(1.0,0,0));
  sphere12VisAtt->SetForceWireframe(true);
  //sphereVisAtt->SetForceSolid(true);
  sphere12VisAtt->SetVisibility(false);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere12Logical->SetVisAttributes(sphere12VisAtt);

  G4VisAttributes* sand10VisAtt = new G4VisAttributes(G4Colour(0.0,1.0,0));
  sand10VisAtt->SetForceWireframe(true);
  sand10VisAtt->SetVisibility(true);
  sand10VisAtt->SetForceSolid(true);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  fSand8Logical->SetVisAttributes(sand10VisAtt);
  
  G4VisAttributes* sphere13VisAtt = new G4VisAttributes(G4Colour(0.0,1.0,0));
  sphere13VisAtt->SetForceWireframe(true);
  //sphereVisAtt->SetForceSolid(true);
  sphere13VisAtt->SetVisibility(false);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere13Logical->SetVisAttributes(sphere13VisAtt);

  G4VisAttributes* sand11VisAtt = new G4VisAttributes(G4Colour(0.0,1.0,1.0));
  sand11VisAtt->SetForceWireframe(true);
  sand11VisAtt->SetVisibility(true);
  sand11VisAtt->SetForceSolid(true);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  fSand9Logical->SetVisAttributes(sand11VisAtt);
  
  G4VisAttributes* sphere14VisAtt = new G4VisAttributes(G4Colour(0.0,1.0,1.0));
  sphere14VisAtt->SetForceWireframe(true);
  //sphereVisAtt->SetForceSolid(true);
  sphere14VisAtt->SetVisibility(false);
  // objectVisAtt->SetForceAuxEdgeVisible(true);     
  fSphere14Logical->SetVisAttributes(sphere14VisAtt);
*/
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PepiDetectorConstruction2::DefineDetectors()
{
  // ========================================
  //                 SCORING
  // ========================================

  G4SDManager* pSDMan = G4SDManager::GetSDMpointer();
  pSDMan->SetVerboseLevel(1);
  
  // - Create 2 filters:
  // Simple particle filter --> Only gamma are detected
  // G4SDParticleFilter* gammaFilter = new G4SDParticleFilter("gammaFilter", "gamma");

  // - Particle filter with energy thresholds --> Only gamma with energy
  // between eKMin and eKMax are detected
  G4double eKMin = 1*keV;
  G4double eKMax = 150*keV;

  G4SDParticleWithEnergyFilter* gammaEKinFilter = new G4SDParticleWithEnergyFilter("gammaEKinFilter",eKMin,eKMax);
  gammaEKinFilter->add("gamma");

  // ========================================
  // - Define a Multifunctional detector 
  // ----> Ideal photon counter 
  // ========================================

  G4MultiFunctionalDetector* pixiRadSD = new G4MultiFunctionalDetector("PixiRadSD");
  pSDMan->AddNewDetector(pixiRadSD);
  SetSensitiveDetector("PixelLV",pixiRadSD);

  // - Ideal photon counter  scores the number of gammas that hit its -Z surface from outside
  // Surface is defined at the -Z surface.
  // Direction                  -Z   +Z
  //   0  IN || OUT            ->|<-  |
  //   1  IN                   ->|    |
  //   2  OUT                    |<-  |
  G4PSFlatSurfaceCurrent* sTotSurfCurrent = new G4PSFlatSurfaceCurrent("TotalSurfCurrent", 1);
  sTotSurfCurrent->SetFilter(gammaEKinFilter);
  sTotSurfCurrent->DivideByArea(false);
  sTotSurfCurrent->Weighted(false);


  pixiRadSD->RegisterPrimitive(sTotSurfCurrent);


  // -----------------------------------------------
  // - Photon counter (Pixirad) with realistic energy response and up to 2 thresholds per pixel
  if(fDetType=="1COL"|| fDetType=="2COL")
  {
      PepiPSPixiRad* sPixiRad = new PepiPSPixiRad("Threshold1", fThreshold1, 
                                                fnPixelsX, fnPixelsY,
                                                fPixelSizeX, fPixelSizeY, fPixelSizeZ, "keV");

      pixiRadSD->RegisterPrimitive(sPixiRad);
      
      if(fDetType=="2COL")
      {
      PepiPSPixiRad* sPixiRad2 = new PepiPSPixiRad("Threshold2", fThreshold2, 
                                                fnPixelsX, fnPixelsY,
                                                fPixelSizeX, fPixelSizeY, fPixelSizeZ, "keV");

      pixiRadSD->RegisterPrimitive(sPixiRad2);
      
      }
  }
  
  // ========================================  
  // - Define a Multifunctional detector 
  // ----> ION CHAMBER to check if there is flux in front of the detector 
  // ========================================

  G4MultiFunctionalDetector* ionChamberSD = new G4MultiFunctionalDetector("IonChamberSD");
  pSDMan->AddNewDetector(ionChamberSD);
  SetSensitiveDetector("IonChamberLV",ionChamberSD);

  // - Ion Chamber scores the number of photons that enter its surface
  // Surface is defined at the -Z surface.
  // Direction                  -Z   +Z
  //   0  IN || OUT            ->|<-  |
  //   1  IN                   ->|    |
  //   2  OUT                    |<-  |
  G4PSFlatSurfaceCurrent* sCurrentIoC = new G4PSFlatSurfaceCurrent("CurrentIoC", 1, "permm2");
  sCurrentIoC->SetFilter(gammaEKinFilter);
  
  ionChamberSD->RegisterPrimitive(sCurrentIoC);

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PepiDetectorConstruction2::SetObjectDetDistance(G4double objectDetDistance)
{
  if(!fConstructed)
  {
    if(objectDetDistance < 10*cm || objectDetDistance > 2*m)
    {
      G4cerr << objectDetDistance << " is out of bounds (Must be > 0.1 m AND < 2 m. - Command is ignored." << G4endl;
    }
    else
    {
      fObjectDetDistance = objectDetDistance;
    }
  }
  else
  {
    G4cerr << "Cannot change geometry after initialization. - Command is ignored." << G4endl;
    return;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
// NEW for PEPI
void PepiDetectorConstruction2::SetSourcePosZ(G4double sourcePosZ)
{
  if(!fConstructed)
  {
    if(sourcePosZ < -2.3/2*m)
    {
      G4cerr << sourcePosZ << "Source is probably outside the World volume - Command is ignored." << G4endl;
    }
    else
    {
      fSourcePosZ = sourcePosZ;
    }
  }
  else
  {
    G4cerr << "Cannot change geometry after initialization. - Command is ignored." << G4endl;
    return;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
// NEW for PEPI
void PepiDetectorConstruction2::SetSrcObjDistance(G4double srcObjDistance)
{
  if(!fConstructed)
  {
    if(srcObjDistance < 0*cm || srcObjDistance > 2.3*m)
    {
      G4cerr << srcObjDistance << "Source object distance is out of bounds (Must be > 0 m AND < World size. - Command is ignored." << G4endl;
    }
    else
    {
      fSrcObjDistance = srcObjDistance;
    }
  }
  else
  {
    G4cerr << "Cannot change geometry after initialization. - Command is ignored." << G4endl;
    return;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void PepiDetectorConstruction2::SetMaskThickness(G4double maskThickness)
{
  if(!fConstructed)
  {
    if(maskThickness < 0*um || maskThickness > 1000*um)
    {
      G4cerr << maskThickness << "Mask thickness is out of bounds (Must be > 0 um AND < 1000 um (default 300 um).- Command is ignored." << G4endl;
    }
    else
    {
      fMaskThickness = maskThickness;
    }
    G4cout <<"The mask thickness is "<< fMaskThickness/um <<"um"<< G4endl;
  }
  else
  {
    G4cerr << "Cannot change geometry after initialization. - Command is ignored." << G4endl;
    return;
  }
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void PepiDetectorConstruction2::SetM2Pitch(G4double m2Pitch)
{
  if(!fConstructed)
  {
    if(m2Pitch < 0*um || m2Pitch > 1000*um)
    {
      G4cerr << m2Pitch << "Mask pitch is out of bounds (Must be > 0 um AND < 1000 um (default 62 um).- Command is ignored." << G4endl;
    }
    else
    {
      fM2Pitch = m2Pitch;
    }
    G4cout <<"The detector mask pitch is "<< fM2Pitch/um <<"um"<< G4endl;
  }
  else
  {
    G4cerr << "Cannot change geometry after initialization. - Command is ignored." << G4endl;
    return;
  }
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void PepiDetectorConstruction2::SetM2Aperture(G4double m2Aperture)
{
  if(!fConstructed)
  {
    if(m2Aperture < 0*um || m2Aperture > fM2Pitch)
    {
      G4cerr << m2Aperture << "Mask aperture is out of bounds (Must be > 0 um AND < pitch (default 15 um).- Command is ignored." << G4endl;
    }
    else
    {
      fM2Aperture = m2Aperture;
    }
    G4cout <<"The detector mask aperture is "<< fM2Aperture/um <<"um"<< G4endl;
  }
  else
  {
    G4cerr << "Cannot change geometry after initialization. - Command is ignored." << G4endl;
    return;
  }
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void PepiDetectorConstruction2::SetThreshold(G4double threshold1, G4double threshold2)
{  
  if(!fConstructed)
  {
    fThreshold1=threshold1;
    fThreshold2=threshold2;
  }
  else
  {
    G4cerr << "Cannot change threshold after inizialization. - Command is ignored." << G4endl;
    return;
  }  
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PepiDetectorConstruction2::SetBidimensionalAcquisition(G4bool bidimensional)
{
  if(fBidimensional) return;

  fBidimensional = bidimensional;
  //G4cout<< bidimensional << G4endl;
  if(!fConstructed) return;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PepiDetectorConstruction2::SetEIMovements(G4double trans, G4double dith, G4double rotAngle)
{
  fTrans    = trans;
  fDith     = dith;
  fRotAngle = rotAngle;

  if(!fConstructed) return;
// Stepping and Dithering of the sample
/* 
 // object 1
  G4ThreeVector position1 = G4ThreeVector(1.5*mm+fTrans+fDith, 0, fSourcePosZ+fSrcObjDistance);
  Move("Trap", fObjectLogical, fObjectPhysical, position1);

 // object 2
  G4ThreeVector position2 = G4ThreeVector(0.*mm+fTrans+fDith, 0, fSourcePosZ+fSrcObjDistance);
  Move("Trap2", fObject2Logical, fObject2Physical, position2);

 // object 3
  G4ThreeVector position3 = G4ThreeVector(-1.5*mm+fTrans+fDith, 0, fSourcePosZ+fSrcObjDistance);
  Move("Trap3", fObject3Logical, fObject3Physical, position3);
*/
  G4cout<<"Sample translated to " << (fTrans+fDith)/um << " um" <<G4endl;
 /* 
  if(fAcquisitionType=="singlemask" || fAcquisitionType=="doublemask")
  {
 // Translation Sample Mask M1 and substrate
  G4double rel_mag = fSrcObjDistance/(fSrcObjDistance-(fMaskThickness+2*fObjSizeR)/2);
  G4ThreeVector position = G4ThreeVector(fTrans/rel_mag, 0, fSourcePosZ+fSrcObjDistance-(fMaskThickness+2*fObjSizeR)/2);
  Move("EnvelopeM1", fEnvelopeM1Logical, fEnvelopeM1Physical, position);
  Move("M1sub", fM1subLogical, fM1subPhysical, position-G4ThreeVector(0,0,fSubThickness/2+fMaskThickness/2));
 // print  
  G4cout<<"Sample Mask translated to " << fTrans/rel_mag/um << " um" <<G4endl;
  }
*/
  G4RunManager::GetRunManager()->GeometryHasBeenModified();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PepiDetectorConstruction2::SetObjectMaterial(G4String materialChoice)
{
  G4Material* pttoMaterial = G4Material::GetMaterial(materialChoice);

  if(pttoMaterial){
    fObjectMaterial = pttoMaterial;
    if(fConstructed) fObjectLogical->SetMaterial(fObjectMaterial);

    G4RunManager::GetRunManager()->GeometryHasBeenModified();
  }
  else G4cerr << materialChoice << " is not defined. - Command is ignored." << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void PepiDetectorConstruction2::SetAcquisitionType(G4String acquisitionType)
{
  fAcquisitionType = acquisitionType;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void PepiDetectorConstruction2::SetDetType(G4String detType)
{
  fDetType = detType;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void PepiDetectorConstruction2::SetCheckOverlaps(G4bool checkOverlaps)
{
  fCheckOverlaps = checkOverlaps;
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
std::vector<double> PepiDetectorConstruction2::LoadDelta(G4String FileName) 
{
G4cout<<"reading delta values from "<< FileName << G4endl;

  std::ifstream myData(FileName, std::ios::binary);
  std::vector<G4double> deltas;
  if(!myData.is_open())//file not open
    {
        G4cout<< FileName << "file does not exist in the specified path" << G4endl;
	return deltas;
    }
  double num = 0.0;
  while (myData >> num){
      deltas.push_back(num);
  }
  return deltas;
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
std::vector<G4ThreeVector> PepiDetectorConstruction2::ReadCoordinates(G4String FileName)
{
G4cout<<"reading positions from"<< FileName <<G4endl;

    std::vector<G4ThreeVector> coordinates;
    std::ifstream myData(FileName, std::ios::binary);
    //std::ifstream infile(FileName);
    if(!myData.is_open()) 
       {
  		G4cout<< FileName << "file does not exist in the specified path" << G4endl;
		return coordinates;       
       }
       
       double x, y, z;
       while (myData >> x >> y >> z) {
            coordinates.push_back(G4ThreeVector(x * mm, y * mm, z * mm));
        }
       
    return coordinates;
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
std::tuple<G4LogicalVolume*,G4VPhysicalVolume*> PepiDetectorConstruction2::CreateMask(G4String name, G4double mag, G4double pitch, G4double aperture, G4ThreeVector position, G4double thickness, G4Material* material, G4LogicalVolume* EnvelopeLogical, G4VPhysicalVolume* EnvelopePhysical)
{

  // - Build the MASK APERTURE UNIT as a Box
  G4Box* MSolid =     new G4Box(name,                               //its name
                          ((pitch-aperture)/mag)/2,   //its size
                          1.1*fPixiRadSizeY/2,
                          thickness/2);

  G4String lvname = name+"LV";                   
  G4LogicalVolume* MLogical = new G4LogicalVolume(MSolid,       //its solid
                                   material,  //its material
                                   lvname);        //its name

  // Build the MASK ENVELOPE 
  G4String envname = "Envelope"+name;                     
  G4Box* EnvelopeSolid =  new G4Box(envname,                  //its name                 
                        (1.1*fPixiRadSizeX/mag)/2,              //its size
                        1.1*fPixiRadSizeY/2,
                        thickness/2);        
  G4String lvenvname = "Envelope"+name+"LV";                         
  EnvelopeLogical =  new G4LogicalVolume(EnvelopeSolid,    //its solid
                                            fWorldMaterial,      //its material
                                            lvenvname);     //its name

  // - Place the physical copies of the mask aperture unit
   G4int copy_no=0;  

    for (G4int iX = 0; iX < int(fnPixelsX*fPixelSizeX/pitch)+2; iX++)
    {
          
      G4double x = + iX*pitch/mag - ((fPixiRadSizeX/mag)/2 + (pitch)/mag + ((pitch)/mag)/2);
      G4double y = 0;
 // G4cout << "position \n" << x <<G4endl;
      
      G4ThreeVector px_position = G4ThreeVector(x, y, 0);
      G4String  name1 = name + "_" + G4UIcommand::ConvertToString(copy_no);

      /*G4VPhysicalVolume* MPhysical =*/  new G4PVPlacement(0,                           //its rotation
                                       px_position,                    //its position
                                       MLogical,                  //its logical volume
                                       name1,                        //its name
                                       EnvelopeLogical,          //its mother volume
                                       false,                       //no boolean operation
                                       copy_no,                     //copy number
                                       fCheckOverlaps);             //checking overlaps 
      copy_no++;                              
    }

  // - Place the Sample Mask Envelope in the World
  EnvelopePhysical = new G4PVPlacement(0,                                                  //its rotation
                                          position,   				      //its position
                                          EnvelopeLogical,                                    //its logical volume
                                          envname,                                       //its name
                                          fWorldLogical,                                      //its mother volume
                                          false,                                              //no boolean operation
                                          0,                                                  //copy number
                                          fCheckOverlaps);                                    //checking overlaps


  G4Region* aRegion = new G4Region(name);
  EnvelopeLogical->SetRegion(aRegion);
  aRegion->AddRootLogicalVolume(EnvelopeLogical);
  
  

  G4VisAttributes* MVisAtt = new G4VisAttributes(G4Colour(0.8,0.6,0.));
//  M1VisAtt->SetForceWireframe(false);
  MVisAtt->SetForceSolid(true);
  MVisAtt->SetVisibility(true);
  // pixelVisAtt->SetForceAuxEdgeVisible(true);
  MLogical->SetVisAttributes(MVisAtt);

  G4VisAttributes* envelopeMVisAtt = new G4VisAttributes(G4Colour(1.0,0.,0.));
//  envelopeM1VisAtt->SetForceWireframe(false);
  envelopeMVisAtt->SetForceSolid(true);
  envelopeMVisAtt->SetVisibility(false);
  envelopeMVisAtt->SetForceAuxEdgeVisible(false);
  EnvelopeLogical->SetVisAttributes(envelopeMVisAtt);
  
  return std::make_tuple(EnvelopeLogical, EnvelopePhysical);  
  
}//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
std::tuple<G4LogicalVolume*,G4VPhysicalVolume*> PepiDetectorConstruction2::CreateSubstrate(G4String name, G4double mag, G4ThreeVector position, G4double thickness, G4Material* material, G4LogicalVolume* SubLogical, G4VPhysicalVolume* SubPhysical)
{

  // - Build the substrate as a box
  G4Box* SubSolid = new G4Box(name,                               //its name
                          (1.2*fPixiRadSizeX/mag)/2,   //its size
                          (1.2*fPixiRadSizeY)/2,
                          thickness/2);
  G4String lvname = name + "LV";
     
  SubLogical = new G4LogicalVolume(SubSolid,       //its solid
                                       material,    //its material
                                       lvname);          //its name
 

  SubPhysical = new G4PVPlacement(0,              //its rotation
                                      position,       //its translation
                                      SubLogical,      //its logical volume
                                      name,              //its name
                                      fWorldLogical,       //its mother volume
                                      false,               //no boolean operation
                                      0,                   //copy number
                                      fCheckOverlaps);     //checking overlaps

  G4VisAttributes* SubVisAtt = new G4VisAttributes(G4Colour(0.5,0.5,0.5));
//  M1subVisAtt->SetForceWireframe(false);
  SubVisAtt->SetForceSolid(true);
  SubVisAtt->SetVisibility(true);
  SubVisAtt->SetForceAuxEdgeVisible(false);
  SubLogical->SetVisAttributes(SubVisAtt);
  
 return std::make_tuple(SubLogical, SubPhysical);
}  
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void PepiDetectorConstruction2::Move(G4String name, G4LogicalVolume* Logical, G4VPhysicalVolume* Physical, G4ThreeVector position)
{
// for lateral translation of masks and sample

  Logical->RemoveDaughter(Physical);
  delete Physical;
  Physical = new G4PVPlacement(0,                                                  //its rotation
                                          position,   				      //its position
                                          Logical,                                    //its logical volume
                                          name,                                       //its name
                                          fWorldLogical,                                      //its mother volume
                                          false,                                              //no boolean operation
                                          0,                                                  //copy number
                                          fCheckOverlaps);                                    //checking overlaps                                
} 
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
