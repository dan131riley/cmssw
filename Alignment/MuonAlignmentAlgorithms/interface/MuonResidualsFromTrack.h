#ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsFromTrack_H
#define Alignment_MuonAlignmentAlgorithms_MuonResidualsFromTrack_H

/** \class MuonResidualsFromTrack
 *  $Id: $
 *  \author J. Pivarski - Texas A&M University <pivarski@physics.tamu.edu>
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
//#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TMatrixDSym.h"
#include "TMatrixD.h"

#include <vector>
#include <map>

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonChamberResidual.h"

class MuonResidualsFromTrack {
public:
  using BuilderToken = edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord>;
  static edm::ESInputTag builderESInputTag();

  // residuals from global muon trajectories
  MuonResidualsFromTrack(edm::ESHandle<TransientTrackingRecHitBuilder> builder,
                         edm::ESHandle<MagneticField> magneticField,
                         edm::ESHandle<GlobalTrackingGeometry> globalGeometry,
                         edm::ESHandle<DetIdAssociator> muonDetIdAssociator_,
                         edm::ESHandle<Propagator> prop,
                         const Trajectory *traj,
                         const reco::Track *recoTrack,
                         AlignableNavigator *navigator,
                         double maxResidual);

  // residuals from tracker muons
  MuonResidualsFromTrack(edm::ESHandle<GlobalTrackingGeometry> globalGeometry,
                         const reco::Muon *recoMuon,
                         AlignableNavigator *navigator,
                         double maxResidual);

  ~MuonResidualsFromTrack();

  void clear();

  const reco::Track *getTrack() { return m_recoTrack; }
  const reco::Muon *getMuon() { return m_recoMuon; }

  int trackerNumHits() const { return m_tracker_numHits; }

  double trackerChi2() const { return m_tracker_chi2; }
  double trackerRedChi2() const;
  double normalizedChi2() const;

  bool contains_TIDTEC() const { return m_contains_TIDTEC; }

  const std::vector<DetId> chamberIds() const { return m_chamberIds; }

  MuonChamberResidual *chamberResidual(DetId chamberId, int type);

  TMatrixDSym covMatrix(DetId chamberId);
  TMatrixDSym corrMatrix(DetId chamberId);
  TMatrixD choleskyCorrMatrix(DetId chamberId);

private:
  //Due to large constructor code, UBSAN with GCC14 hangs and never returns
  //Moving large constructor code to a private function fixes this issue
  void init(edm::ESHandle<TransientTrackingRecHitBuilder> builder,
            edm::ESHandle<MagneticField> magneticField,
            edm::ESHandle<GlobalTrackingGeometry> globalGeometry,
            edm::ESHandle<DetIdAssociator> muonDetIdAssociator_,
            edm::ESHandle<Propagator> prop,
            const Trajectory *traj,
            const reco::Track *recoTrack,
            AlignableNavigator *navigator,
            double maxResidual);

  TrajectoryStateCombiner m_tsoscomb;

  int m_tracker_numHits;
  double m_tracker_chi2;
  bool m_contains_TIDTEC;

  std::vector<DetId> m_chamberIds;
  std::map<DetId, MuonChamberResidual *> m_dt13, m_dt2, m_csc;
  std::map<DetId, TMatrixDSym> m_trkCovMatrix;

  void addTrkCovMatrix(DetId, TrajectoryStateOnSurface &);

  // pointer to its track
  const reco::Track *m_recoTrack;

  // track muon
  const reco::Muon *m_recoMuon;
};

#endif  // Alignment_MuonAlignmentAlgorithms_MuonResidualsFromTrack_H
