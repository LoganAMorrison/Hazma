      SUBROUTINE RAMBO(N,ET,XM,P,WT)
C------------------------------------------------------
C
C                       RAMBO
C
C    RA(NDOM)  M(OMENTA)  B(EAUTIFULLY)  O(RGANIZED)
C
C    A DEMOCRATIC MULTI-PARTICLE PHASE SPACE GENERATOR
C    AUTHORS:  S.D. ELLIS,  R. KLEISS,  W.J. STIRLING
C    THIS IS VERSION 1.0 -  WRITTEN BY R. KLEISS
C
C    MODIFIED SLIGHTLY BY IAN HINCHLIFFE
C
C    N  = NUMBER OF PARTICLES (>1, IN THIS VERSION <101)
C    ET = TOTAL CENTRE-OF-MASS ENERGY
C    XM = PARTICLE MASSES ( DIM=100 )
C    P  = PARTICLE MOMENTA ( DIM=(4,100) )
C    WT = WEIGHT OF THE EVENT
C
C------------------------------------------------------
      IMPLICIT NONE
      REAL*8 Z,XM,P,Q,R,B,XM2,E,V,WT,ACC,PO2LOG,XMT,ET,RN,XMAX
      REAL*8 C,S,F,RANDOM1,RMAS,A,X,G,BQ,ACCU,WT2,WT3,WTM,P2
      REAL*8 F0,G0,X2,PI
      INTEGER*4 IWARN,IBEGIN,ITMAX,K,N,NM,ITER,I
      DIMENSION XM(100),P(4,100),Q(4,100),R(4),
     .   B(3),P2(100),XM2(100),E(100),V(100),IWARN(5),Z(100)
      DATA ACC/1.D-14/,ITMAX/6/,IBEGIN/0/,IWARN/5*0/
      DATA PO2LOG/4.515827D-1/,PI/3.14159/
C PO2LOG IS LOG(PI/2)
C
C INITIALIZATION STEP: FACTORIALS FOR THE PHASE SPACE WEIGHT
      IF(IBEGIN.EQ.0) THEN
        IBEGIN=1
        Z(2)=PO2LOG
        DO 33 K=3,100
          Z(K)=Z(K-1)+PO2LOG-2.*LOG(FLOAT(K-2))
   33   CONTINUE
        DO 35 K=3,100
          Z(K)=(Z(K)-LOG(FLOAT(K-1)))
   35   CONTINUE
C CHECK ON THE NUMBER OF PARTICLES
      ELSE IF(N.LT.2.OR.N.GT.100) THEN
        PRINT 1001,N
        STOP
      END IF
C
C CHECK WHETHER TOTAL ENERGY IS SUFFICIENT; COUNT NONZERO MASSES
      XMT=0.D0
      NM=0
      DO 17 I=1,N
        IF(XM(I).NE.0.) NM=NM+1
        XMT=XMT+ABS(XM(I))
   17 CONTINUE
      IF(XMT.GT.ET) THEN
        PRINT 1002,XMT,ET
        STOP
      END IF
C
C THE PARAMETER VALUES ARE NOW ACCEPTED
C
C GENERATE N MASSLESS MOMENTA IN INFINITE PHASE SPACE
c      write(*,*) 'rambo--start of do 27 loop'
      DO 27 I=1,N
c        write(*,*) 'rambo--calling rn 1'
        C=2.D0*RN(1)-1.
c      write(*,*) 'rambo--rn 1', (C+1.d0)/2.d0
        S=SQRT(1.-C*C)
c        write(*,*) 'rambo--calling rn'
        F=2.*PI*RN(2)
c        write(*,*) 'rambo--rn(2)', F/(2.*PI)
        RANDOM1=RN(3)*RN(4)
        IF(RANDOM1.EQ.0) THEN
          RANDOM1=RN(3)*RN(4)
        END IF
        Q(4,I)=-LOG(RANDOM1)
        Q(3,I)=Q(4,I)*C
        Q(2,I)=Q(4,I)*S*COS(F)
        Q(1,I)=Q(4,I)*S*SIN(F)
   27 CONTINUE
C
C CALCULATE THE PARAMETERS OF THE CONFORMAL TRANSFORMATION
      DO 93 I=1,4
        R(I)=0.D0
   93 CONTINUE
      DO 95 I=1,N
        DO 97 K=1,4
          R(K)=R(K)+Q(K,I)
   97   CONTINUE
   95 CONTINUE
      RMAS=SQRT(R(4)**2-R(3)**2-R(2)**2-R(1)**2)
      DO 98 K=1,3
        B(K)=-R(K)/RMAS
   98 CONTINUE
      G=R(4)/RMAS
      A=1.D0/(1.+G)
      X=ET/RMAS
C
C TRANSFORM THE Q'S CONFORMALLY INTO THE P'S
      DO 43 I=1,N
        BQ=B(1)*Q(1,I)+B(2)*Q(2,I)+B(3)*Q(3,I)
        DO 45 K=1,3
          P(K,I)=X*(Q(K,I)+B(K)*(Q(4,I)+A*BQ))
   45   CONTINUE
        P(4,I)=X*(G*Q(4,I)+BQ)
   43 CONTINUE
C
C CALCULATE WEIGHT AND POSSIBLE WARNINGS
      WT=PO2LOG
      IF(N.NE.2) WT=(2.*N-4.)*LOG(ET)+Z(N)
      IF(WT.LT.-180.) THEN
        IF(IWARN(1).LE.5) PRINT 1004,WT
        IWARN(1)=IWARN(1)+1
      ELSE IF(WT.GT. 174.) THEN
        IF(IWARN(2).LE.5) PRINT 1005,WT
        IWARN(2)=IWARN(2)+1
C
C RETURN FOR WEIGHTED MASSLESS MOMENTA
      ELSE IF(NM.EQ.0) THEN
        WT=DEXP(WT)
        RETURN
      END IF
C
C MASSIVE PARTICLES: RESCALE THE MOMENTA BY A FACTOR X
      XMAX=SQRT(1.-(XMT/ET)**2)
      DO 63 I=1,N
        XM2(I)=XM(I)**2
        P2(I)=P(4,I)**2
   63 CONTINUE
      ITER=0
      X=XMAX
      ACCU=ET*ACC
      DO WHILE (ITER.LE.ITMAX)
        F0=-ET
        G0=0.D0
        X2=X*X
        DO 65 I=1,N
          E(I)=SQRT(XM2(I)+X2*P2(I))
          F0=F0+E(I)
          G0=G0+P2(I)/E(I)
   65   CONTINUE
        IF(ABS(F0).GT.ACCU.AND.ITER.LE.ITMAX) THEN
          ITER=ITER+1
          X=X-F0/(X*G0)
        ELSE IF(ABS(F0).LE.ACCU) THEN
          ITER=ITMAX+1
        ELSE
          PRINT 1006,ITMAX
        END IF
      END DO
      DO 67 I=1,N
        V(I)=X*P(4,I)
        DO 69 K=1,3
          P(K,I)=X*P(K,I)
   69   CONTINUE
        P(4,I)=E(I)
   67 CONTINUE
C
C CALCULATE THE MASS-EFFECT WEIGHT FACTOR
      WT2=1.D0
      WT3=0.D0
      DO 73 I=1,N
        WT2=WT2*V(I)/E(I)
        WT3=WT3+V(I)**2/E(I)
   73 CONTINUE
      WTM=(2.*N-3.)*LOG(X)+LOG(WT2/WT3*ET)
C
C RETURN FOR  WEIGHTED MASSIVE MOMENTA
      WT=WT+WTM
      IF(WT.LT.-180.) THEN
        IF(IWARN(3).LE.5) PRINT 1004,WT
        IWARN(3)=IWARN(3)+1
      ELSE IF(WT.GT. 174.) THEN
        IF(IWARN(4).LE.5) PRINT 1005,WT
        IWARN(4)=IWARN(4)+1
      END IF
      WT=DEXP(WT)
      RETURN
C
 1001 FORMAT(' RAMBO FAILS: # OF PARTICLES =',I5,' IS NOT ALLOWED')
 1002 FORMAT(' RAMBO FAILS: TOTAL MASS =',D15.6,' IS NOT',
     .  ' SMALLER THAN TOTAL ENERGY =',D15.6)
 1004 FORMAT(' RAMBO WARNS: WEIGHT = EXP(',F20.9,') MAY UNDERFLOW')
 1005 FORMAT(' RAMBO WARNS: WEIGHT = EXP(',F20.9,') MAY  OVERFLOW')
 1006 FORMAT(' RAMBO WARNS:',I3,' ITERATIONS DID NOT GIVE THE',
     .  ' DESIRED ACCURACY =',D15.6)
C******************************************************
      END
c---------------------------------------------------------------------------
c      double precision function rn(idum)
c      integer*4 idum
c      real dum,myran_sing
c      rn = dble(myran_sing)
c      return
c      end
c-------------------------------------------------------------------------
      

