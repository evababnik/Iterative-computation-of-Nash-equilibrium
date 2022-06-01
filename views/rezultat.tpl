%rebase('baza.tpl')
<h1> Izračun Nashevega ravnovesja in vrednosti igre<\h1>

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-0pky">Metoda simpleksov</th>
    <th class="tg-0pky">Metoda I</th>
    <th class="tg-0pky">Metoda II</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">Vrednost igre</td>
    <td class="tg-0pky">{{vred_simpleks}}</td>
    <td class="tg-0pky">{{vred_1}}</td>
    <td class="tg-0pky">{{vred_2}}</td>
  </tr>
  <tr>
    <td class="tg-0pky">Optimalna strategija 1. igralca</td>
    <td class="tg-0pky">{{x_simpleks}}</td>
    <td class="tg-0pky">{{x_1}}</td>
    <td class="tg-0pky">{{x_2}}</td>
  </tr>
  <tr>
    <td class="tg-0pky">Optimalna strategija 2. igralca</td>
    <td class="tg-0pky">{{y_simpleks}}</td>
    <td class="tg-0pky">{{y_1}}</td>
    <td class="tg-0pky">{{y_2}}</td>
  </tr>
  <tr>
    <td class="tg-0pky">Čas</td>
    <td class="tg-0pky">{{cas_simpleks}}</td>
    <td class="tg-0pky">{{cas_1}}</td>
    <td class="tg-0pky">{{cas_2}}</td>
  </tr>
  <tr>
    <td class="tg-0pky">Število korakov</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">{{koraki_1}}</td>
    <td class="tg-0pky">{{koraki_2}}</td>
  </tr>
</tbody>
</table>

<form action="/zacetek/" method="GET">
           <button type="submit" class="btn btn-secondary btn-lg">Reši novo matrično igro!</button>
</form>